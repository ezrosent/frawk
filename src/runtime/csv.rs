#[allow(unused)]
/// A WIP port (with modifications) of geofflangdale/simdcsv.

#[derive(Default)]
struct Offsets {
    // NB We're using u64s to potentially handle huge input streams.
    // An alternative option would be to save lines and fields in separate vectors, but this is
    // more efficient for iteration
    fields: Vec<u64>,
}

// TODO: store what character we have in the high bits? or maybe just use prefetching to read the
//       proper locations.
// TODO: materialize
//  iterate over fields:
//    Keep track of previous location, initialize at 0. Iterate over `fields` and case-analyze the
//    pointed-to character. A few of these will rely on having 0s padding the end.
//    if we have a comma, append slice of [prev, loc of comma) to current record.
//    if we have a newline, append slice of [prev, loc of comma) to current record and append
//      record to output.
//    if we have a quote, check ahead one and if it's a quote append the quote, otherwise do
//      nothing.
//    if we have a backslash, check ahead and escape appropriately, consider about whether to
//      return an error or just append the slash if we don't see it.

#[cfg(target_arch = "x86_64")]
#[allow(unused)]
mod avx2 {
    use super::Offsets;
    // This is in a large part based on geofflangdale/simdcsv, which is an adaptation of some of
    // the techniques from the simdjson paper (by that paper's authors, it appears) to CSV parsing.
    use std::arch::x86_64::*;
    #[derive(Copy, Clone)]
    struct Input {
        lo: __m256i,
        hi: __m256i,
    }

    const VEC_BYTES: usize = 32;
    const INPUT_SIZE: usize = VEC_BYTES * 2;

    #[inline(always)]
    unsafe fn fill_input(bptr: *const u8) -> Input {
        Input {
            lo: _mm256_loadu_si256(bptr as *const _),
            hi: _mm256_loadu_si256(bptr.offset(VEC_BYTES as isize) as *const _),
        }
    }

    #[inline(always)]
    unsafe fn cmp_mask_against_input(inp: Input, m: u8) -> u64 {
        // Load the mask into all lanes.
        let mask = _mm256_set1_epi8(m as i8);
        // Compare against lo and hi. Store them in a single u64.
        let cmp_res_0 = _mm256_cmpeq_epi8(inp.lo, mask);
        let res_0 = _mm256_movemask_epi8(cmp_res_0) as u32 as u64;
        let cmp_res_1 = _mm256_cmpeq_epi8(inp.hi, mask);
        let res_1 = _mm256_movemask_epi8(cmp_res_1) as u64;
        res_0 | (res_1 << 32)
    }

    #[inline(always)]
    unsafe fn find_quote_mask(
        inp: Input,
        prev_iter_inside_quote: &mut u64,
    ) -> (/*inside quotes*/ u64, /*quote locations*/ u64) {
        // This is about finding a mask that has 1s for all characters inside a quoted pair, plus
        // the starting quote, but not the ending one. For example:
        // [unquoted text "quoted text"]
        // has the mask
        // [000000000000001111111111110]
        // We will use this mask to avoid splitting on commas that are inside a quoted field. We
        // start by generating a mask for all the quote characters appearing in the string.
        let quote_bits = cmp_mask_against_input(inp, '"' as u8);
        // Then we pull this trick from the simdjson paper. Lets use the example from the comments
        // above:
        // [unquoted text "quoted text"]
        // quoted_bits is going to be
        // [000000000000001000000000001]
        // We want to fill anything between the ones with ones. One way to do this is to replace
        // bit i with the prefix sum mod 2 (ie xor) of all the bits up to that point. We could do
        // this by repeated squaring, but intel actually just gives us this instruction for free:
        // it's "carryless multiplication" by all 1s. Caryless multiplication takes the prefix xor
        // of a each bit up to bit i, and-ed by corresponding bits in the other operand; anding
        // with all 1s gives us the sum we want.
        let mut quote_mask =
            _mm_cvtsi128_si64(/* grab the lower 64 bits */ _mm_clmulepi64_si128(
                /*pad with zeros*/ _mm_set_epi64x(0i64, quote_bits as i64),
                /*all 1s*/ _mm_set1_epi8(!0),
                0,
            )) as u64;
        let prev = *prev_iter_inside_quote;
        // We need to invert the mask if we started off inside a quote.
        quote_mask ^= prev;
        // We want all 1s if we ended in a quote, all zeros if not
        *prev_iter_inside_quote = (quote_mask as i64).wrapping_shr(63) as u64;
        (quote_mask, quote_bits)
    }

    #[inline(always)]
    unsafe fn flatten_bits(base_p: *mut u64, start_offset: &mut u64, start_ix: u64, mut bits: u64) {
        // We want to write indexes of set bits into the array base_p.
        // start_offset is the initial offset into base_p where we start writing.
        // start_ix is the index into the corpus pointed to by the initial bits of `bits`.
        if bits == 0 {
            return;
        }
        let count = bits.count_ones();
        let next_start = (*start_offset) + count as u64;
        // Unroll the loop for the first 8 bits. As in simdjson, we are potentially "overwriting"
        // here. We do not know if we have 8 bits to write, but we are writing them anyway! This is
        // "safe" so long as we have enough space in the array pointed to by base_p because:
        // * Excessive writes will only write a zero
        // * we will set next start to point to the right location.
        // Why do the extra work? We want to avoid extra branches.
        macro_rules! write_offset_inner {
            ($ix:expr) => {
                *base_p.offset(*start_offset as isize + $ix) =
                    start_ix + bits.trailing_zeros() as u64;
                bits = bits & bits.wrapping_sub(1);
            };
        }
        macro_rules! write_offsets {
            ($($ix:expr),*) => { $(write_offset_inner!($ix);)* };
        }
        write_offsets!(0, 1, 2, 3, 4, 5, 6, 7);
        // Similarlty for bits 8->16
        if count > 8 {
            write_offsets!(8, 9, 10, 11, 12, 13, 14, 15);
        }
        // Just do a loop for the last 48 bits.
        if count > 16 {
            *start_offset += 16;
            loop {
                write_offsets!(0);
                if bits == 0 {
                    break;
                }
                *start_offset += 1;
            }
        }
        *start_offset = next_start;
    }

    // unsafe because `buf` must have padding 32 bytes past the end, and has to have its length be
    // a multiple of 64.
    unsafe fn find_indexes(
        buf: &[u8],
        offsets: &mut Offsets,
        mut prev_iter_inside_quote: u64, /*start at 0*/
        mut prev_iter_cr_end: u64,       /*start at 0*/
    ) -> (u64, u64) {
        // debug_assert_eq!(buf.len() % INPUT_SIZE, 0);
        offsets.fields.clear();
        // This may cause us to overuse memory, but it's a safe upper bound and the plan is to
        // reuse this across different chunks.
        offsets.fields.reserve(buf.len());
        let buf_ptr = buf.as_ptr();
        let len = buf.len();
        let len_minus_64 = len.saturating_sub(INPUT_SIZE);
        let mut ix = 0;
        let base_ptr: *mut u64 = offsets.fields.get_unchecked_mut(0);
        let mut base = 0;

        // For ... reasons (better pipelining? better cache behavior?) we decode in blocks.
        const BUFFER_SIZE: usize = 4;
        macro_rules! iterate {
            ($buf:expr) => {{
                std::intrinsics::prefetch_read_data($buf.offset(128), 3);
                // find commas not inside quotes
                let inp = fill_input($buf);
                let (quote_mask, quote_locs) = find_quote_mask(inp, &mut prev_iter_inside_quote);
                let sep = cmp_mask_against_input(inp, ',' as u8);
                let esc = cmp_mask_against_input(inp, '\\' as u8);

                let cr = cmp_mask_against_input(inp, 0x0d);
                let cr_adjusted = cr.wrapping_shl(1) | prev_iter_cr_end;
                let lf = cmp_mask_against_input(inp, 0x0a);
                // Allow for either \r\n or \n.
                let end = (lf & cr_adjusted) | lf;
                prev_iter_cr_end = cr.wrapping_shr(63);
                (((end | sep) & !quote_mask) | (esc & quote_mask) | quote_locs)
            }};
        }
        if len_minus_64 > INPUT_SIZE * BUFFER_SIZE {
            let mut fields = [0u64; BUFFER_SIZE];
            while ix < len_minus_64 - INPUT_SIZE * BUFFER_SIZE + 1 {
                for b in 0..BUFFER_SIZE {
                    fields[b] = iterate!(buf_ptr.offset((INPUT_SIZE * b + ix) as isize));
                }
                for b in 0..BUFFER_SIZE {
                    let internal_ix = INPUT_SIZE * b + ix;
                    flatten_bits(base_ptr, &mut base, internal_ix as u64, fields[b]);
                }
                ix += INPUT_SIZE * BUFFER_SIZE;
            }
        }
        // Do an unbuffered version for the remaining data
        while ix < len_minus_64 {
            let field_sep = iterate!(buf_ptr.offset(ix as isize));
            flatten_bits(base_ptr, &mut base, ix as u64, field_sep);
            ix += INPUT_SIZE;
        }
        // For any text that remains, just copy the results to the stack with some padding and do one more iteration.
        let remaining = len - ix;
        if remaining > 0 {
            let mut rest = [0u8; INPUT_SIZE];
            std::ptr::copy_nonoverlapping(
                buf_ptr.offset(ix as isize),
                rest.as_mut_ptr(),
                remaining,
            );
            let field_sep = iterate!(rest.as_mut_ptr());
            flatten_bits(base_ptr, &mut base, ix as u64, field_sep);
        }
        offsets.fields.set_len(base as usize);
        (prev_iter_inside_quote, prev_iter_cr_end)
    }
    #[cfg(test)]
    mod tests {
        use super::*;
        #[test]
        fn basic_impl() {
            let text: &'static str = r#"This,is,"a line with a quoted, comma",and
unquoted,commas,"as well, including some long ones", and there we have it."#;
            let mut mem: Vec<u8> = text.as_bytes().iter().cloned().collect();
            mem.reserve(32);
            let mut offsets: Offsets = Default::default();
            let (in_quote, in_cr) = unsafe { find_indexes(&mem[..], &mut offsets, 0, 0) };
            assert_eq!(in_quote, 0);
            assert_eq!(in_cr, 0);
            assert_eq!(
                &offsets.fields[..],
                &[4, 7, 8, 36, 37, 41, 50, 57, 58, 92, 93],
                "offset_fields={:?}",
                offsets
                    .fields
                    .iter()
                    .cloned()
                    .map(|x| (x, mem[x as usize] as char))
                    .collect::<Vec<_>>()
            );
        }
    }
}
