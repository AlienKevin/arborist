fn main() {
    let path = "./dom_query";
    let lib = "dom_query";

    println!("cargo:rustc-link-search=native={}", path);
    println!("cargo:rustc-link-lib=static={}", lib);
    // println!("cargo:rustc-flags=-l framework=CoreFoundation -l framework=Security");
}
