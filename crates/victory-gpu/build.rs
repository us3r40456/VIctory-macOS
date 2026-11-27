use std::{env, path::PathBuf, process::Command};

// Only a build with VICTORY_HELPER_DXC set runs the compiler. CI never sets it,
// so the DXC path is not built or tested there: it is checked by hand with the
// ignored test `hardware_uses_precompiled_clamp` (see docs/macos.md).
fn main() {
    println!("cargo:rustc-check-cfg=cfg(victory_dxc_clamp)");
    println!("cargo:rustc-check-cfg=cfg(victory_dxc_detile)");
    println!("cargo:rerun-if-env-changed=VICTORY_HELPER_DXC");
    println!("cargo:rerun-if-env-changed=DXC_EXE");
    println!("cargo:rerun-if-changed=src/indirect-clamp.hlsl");
    println!("cargo:rerun-if-changed=src/gpudetile.hlsl");
    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") {
        return;
    }
    let Some(compiler) = env::var_os("VICTORY_HELPER_DXC") else {
        return;
    };
    let compiler = PathBuf::from(compiler);
    if compiler.is_file() {
        println!("cargo:rerun-if-changed={}", compiler.display());
    }
    if let Some(exe) = env::var_os("DXC_EXE") {
        println!("cargo:rerun-if-changed={}", PathBuf::from(exe).display());
    }
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let status = Command::new(&compiler)
        .args(["-T", "cs_6_0", "-E", "main", "-Fo"])
        .arg(out_dir.join("indirect-clamp.dxil"))
        .arg("src/indirect-clamp.hlsl")
        .status()
        .expect("start the configured DXC compiler");
    assert!(
        status.success(),
        "DXC failed for the indirect argument clamp"
    );
    println!("cargo:rustc-cfg=victory_dxc_clamp");

    let detile_status = Command::new(&compiler)
        .args(["-T", "cs_6_0", "-E", "main", "-Fo"])
        .arg(out_dir.join("gpudetile.dxil"))
        .arg("src/gpudetile.hlsl")
        .status()
        .expect("start the configured DXC compiler for gpudetile");
    assert!(
        detile_status.success(),
        "DXC failed for the GPU texture detiler"
    );
    println!("cargo:rustc-cfg=victory_dxc_detile");
}
