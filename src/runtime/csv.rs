/// A WIP port (with modifications) of the geofflangdale/simdcsv.

struct Offsets {
    // NB We're using u64s to potentially handle huge input streams.
    // An alternative option would be to save lines and fields in separate vectors, but this is
    // more efficient for iteration
    // TODO: The high bit indicates if the next field has a quote in it (used for escaping)
    fields: Vec<u64>,
}

// TODO: handle more escaping constructs?
// TODO: look into bug on simdcsv repo
// TODO: look into SIMDfied escaping

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use super::Offsets;
    // This is in a large part based on geofflangdale/simdcsv, which is an adaptation of some of
    // the techniques from the simdjson paper (by that paper's authors, it appears) to CSV parsing.
    use std::arch::x86_64::*;
    struct Input {
        lo: __m256i,
        hi: __m256i,
    }

    const VEC_BYTES: usize = 32;

    #[inline(always)]
    unsafe fn fill_input(bs: &[u8]) -> Input {
        debug_assert!(bs.len() >= VEC_BYTES * 2);
        Input {
            lo: _mm256_loadu_si256(bs.as_ptr() as *const _),
            hi: _mm256_loadu_si256(bs.as_ptr().offset(VEC_BYTES as isize) as *const _),
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
    unsafe fn find_quote_mask(inp: Input, prev_iter_inside_quote: &mut u64) -> u64 {
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
        *prev_iter_inside_quote = (quote_mask as i64).wrapping_shr(64) as u64;
        quote_mask
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
            ($($ix:expr),*) => { $( write_offset_inner!($ix);)* };
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

    // unsafe because `buf` must have padding 32 bytes past the end.
    unsafe fn find_indexes(
        buf: &[u8],
        offsets: &mut Offsets,
        prev_iter_inside_quote: u64, /* start at 0*/
        prev_iter_cr_end: u64,       /*start at 0*/
    ) -> Option<Offsets> {
        offsets.fields.clear();
        // This may cause us to overuse memory, but it's a safe upper bound and the plan is to
        // reuse this across different chunks.
        offsets.fields.reserve(buf.len());
        unimplemented!()
    }
}
