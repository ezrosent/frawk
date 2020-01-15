use crate::common::raw_guard;
use crate::libc::c_char;
use crate::llvm_sys as llvm;
use llvm::{
    analysis::{LLVMVerifierFailureAction, LLVMVerifyModule},
    prelude::LLVMValueRef,
    LLVMLinkage,
};

use std::ffi::CStr;
use std::mem::{self, MaybeUninit};
use std::ptr;

// TODO make sure we use this function in main, otherwise linker may decide to get rid of it.
#[no_mangle]
pub extern "C" fn __test_print() {
    println!("hello! this is rust code called from llvm");
}

pub unsafe fn test_codegen() {
    if llvm::support::LLVMLoadLibraryPermanently(ptr::null()) != 0 {
        panic!("failed to load in-process library");
    }
    use llvm::core::*;
    use llvm::execution_engine::*;
    use llvm::target::*;
    // TODO:
    // LLVM boilerplate
    //   * figure out issues with module verification.
    // Compilation metadata
    //   * For each bytecode function have a set of (type, register) pairs and for each one
    //   indicate
    //      - if it is local  (normal local variable)
    //      - if it is global (passed as additional parameter)
    //      - if it is a return (ignored)
    //   * For each bytecode function, indicate how many arguments it has, and of what type args
    //     and returns are (we should already have this lying around)
    //   * For each bytecode function, have a mapping from Call and Return to arguments and
    //     returns. Call mappings should also include the register (i.e. local or global variable,
    //     or return) in which to store the result.
    //   * Keys for all these mappings can be *const pointers. That should simplify a number of
    //     things. For one, it means we wont have to alter what bytecode does.
    // Runtime
    //   * Figure out extern-C-able versions of the runtime.
    //   * Make sure main uses all functions somehow.
    //   * Most of these are simple, but some things like "moving a string" could be tougher.
    //   * They all may require a "pointer to the runtime" passed in to handle the regex maps, etc.
    //   => We can put the string table in there.
    //   * Make sure to look up how to convert between ints and strings in LLVM.
    // Codegen
    //   * With all the metadata in place, we can do kaleidoscope chapters 5, 7 to implement
    //     everything we need?
    //   * Functions get relevant globals as arguments. Ints and Floats do just
    //     fine, though we still have to figure out that the plan is for Maps.
    //     Globals and locals are aloca'd in the entry block (of main only, for globals).
    //   * We need to figure out what to do about Str.
    //      - We may need a custom Rc that we can store in a pointer (having the ref-count "one
    //        word back" or some-such; then expose everything using that).
    //      - We could store all strings as offsets into a vector (a pointer to which we pass to
    //        every function). Then string indexes could be normal u64s, and all string functions
    //        could take the pointer as well.
    //        > It adds an extra layer of indirection
    //        > _but_ so does moving Rc to the toplevel, and many of the string operations are
    //          fairly heavy-duty.
    //        > This may be the best route.

    // Shared data-structures
    let ctx = LLVMContextCreate();
    let module = raw_guard(
        LLVMModuleCreateWithNameInContext(c_str!("main"), ctx),
        LLVMDisposeModule,
    );
    let builder = raw_guard(LLVMCreateBuilderInContext(ctx), LLVMDisposeBuilder);
    // Jit-specific setup
    LLVM_InitializeNativeTarget();
    LLVM_InitializeNativeAsmPrinter();
    LLVMLinkInMCJIT();
    let mut maybe_engine = MaybeUninit::<LLVMExecutionEngineRef>::uninit();
    let mut err: *mut c_char = ptr::null_mut();
    if LLVMCreateExecutionEngineForModule(maybe_engine.as_mut_ptr(), *module, &mut err) != 0 {
        // NB: In general, want to LLVMDisposeMessage if we weren't just going to crash.
        panic!(
            "failed to create program: {}",
            CStr::from_ptr(err).to_str().unwrap()
        );
    }
    let engine = maybe_engine.assume_init();
    let pass_manager = raw_guard(
        LLVMCreateFunctionPassManagerForModule(*module),
        LLVMDisposePassManager,
    );
    // Take some passes present in most of the tutorials
    {
        use llvm::transforms::scalar::*;
        llvm::transforms::util::LLVMAddPromoteMemoryToRegisterPass(*pass_manager);
        LLVMAddConstantPropagationPass(*pass_manager);
        LLVMAddInstructionCombiningPass(*pass_manager);
        LLVMAddReassociatePass(*pass_manager);
        LLVMAddGVNPass(*pass_manager);
        LLVMAddCFGSimplificationPass(*pass_manager);
        LLVMInitializeFunctionPassManager(*pass_manager);
    }

    // Code generation for __test_print
    let testprint = {
        let testprint_type = LLVMFunctionType(LLVMVoidType(), ptr::null_mut(), 0, 0);
        let tp = LLVMAddFunction(*module, c_str!("__test_print"), testprint_type);
        LLVMSetLinkage(tp, LLVMLinkage::LLVMExternalLinkage);
        tp
    };

    // Code generation for main
    let i64_type = LLVMInt64TypeInContext(ctx);
    let func_ty = LLVMFunctionType(i64_type, ptr::null_mut(), 0, /*is_var_arg=*/ 0);
    let func = LLVMAddFunction(*module, c_str!("main"), func_ty);
    LLVMSetLinkage(func, LLVMLinkage::LLVMExternalLinkage);
    let block = LLVMAppendBasicBlockInContext(ctx, func, c_str!(""));
    LLVMPositionBuilderAtEnd(*builder, block);
    let _ = LLVMBuildCall(*builder, testprint, ptr::null_mut(), 0, c_str!(""));
    LLVMBuildRet(*builder, LLVMConstInt(i64_type, 2, /*sign_extend=*/ 1));
    LLVMRunFunctionPassManager(*pass_manager, func);
    // LLVMVerifyModule(
    //     *module,
    //     LLVMVerifierFailureAction::LLVMAbortProcessAction,
    //     &mut err,
    // );

    // Now, get the code and go!
    let func_addr = LLVMGetFunctionAddress(engine, c_str!("main"));
    if func_addr == 0 {
        panic!("main function is just null!");
    }
    let jitted_func = mem::transmute::<u64, extern "C" fn() -> i64>(func_addr);
    println!("running jitted code");
    LLVMDumpModule(*module);
    let res = jitted_func();
    println!("result={}", res);
    // LLVMBuildCall
}
