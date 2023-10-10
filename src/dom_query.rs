use std::ffi::CString;

use crate::{
    ffi::{GoSlice, GoString, LoadDoc, QuerySelector, QuerySelectors, SetupBenchmark},
    selector::Index,
};

pub fn setup_benchmark(name: &str) {
    let c_str = CString::new(name).expect("CString::new failed in setup_benchmark");
    // TODO: Confirm that c_str is freed by go's GC
    let c_str = std::mem::ManuallyDrop::new(c_str);
    let ptr = c_str.as_ptr();
    let go_string = GoString {
        p: ptr,
        n: c_str.as_bytes().len() as isize,
    };
    unsafe { SetupBenchmark(go_string) };
}

pub fn load_doc(id: Index) {
    unsafe { LoadDoc(id) };
}

pub fn query(dom_trace: Index, selectors: Vec<String>) -> bool {
    load_doc(dom_trace + 1);

    if selectors.len() == 1 {
        query_one(selectors.first().unwrap())
    } else {
        query_many(selectors)
    }
}

fn query_one(selector: &str) -> bool {
    let c_str = CString::new(selector).expect("CString::new failed in query_one");
    let ptr = c_str.as_ptr();
    let go_string = GoString {
        p: ptr,
        n: c_str.as_bytes().len() as isize,
    };
    unsafe { QuerySelector(go_string) != 0 }
}

fn query_many(selectors: Vec<String>) -> bool {
    let mut go_slices: Vec<GoString> = Vec::with_capacity(selectors.len());
    let mut c_strings: Vec<CString> = Vec::with_capacity(selectors.len());

    for selector in selectors {
        let c_str = CString::new(selector.as_str()).expect("CString::new failed in query_all");
        let go_string = GoString {
            p: c_str.as_ptr(),
            n: c_str.as_bytes().len() as isize,
        };
        go_slices.push(go_string);
        c_strings.push(c_str);
    }

    // Now, the c_strings vector owns the CString values and will drop them when it goes out of scope.
    // The pointers in the go_slices vector will remain valid as long as c_strings is not dropped.

    let go_slice = GoSlice {
        data: go_slices.as_mut_ptr() as *mut std::ffi::c_void,
        len: go_slices.len() as i64,
        cap: go_slices.capacity() as i64,
    };

    let result = unsafe { QuerySelectors(go_slice) != 0 };

    result
}
