//! The GLES backend's promises, as the phones without Vulkan need them.
//!
//! GLES 3.0 phones (Redmi 12C, Infinix Hot 30, Galaxy A05 / Mali-G57; Redmi
//! Note 11, POCO C65 / Adreno 610; Realme C11, C21, Narzo 50i / PowerVR
//! GE8322; 3-4 GB of RAM) have no Vulkan, so Horizon renders through wgpu's
//! GLES backend: naga writes GLSL ES, a stage binds at most
//! [`MAX_SAMPLED_TEXTURES_PER_STAGE`] sampled textures, and every pack —
//! builtin or player-made — has to translate at ES 3.0.
//!
//! This file holds the promises the renderer depends on and the naga fixes
//! they rest on: the GLSL feature scan counting only what the entry point
//! can reach (a plain 2D pack must not be asked for `samplerCubeArray`),
//! depth textures becoming shadow samplers only where they are really
//! sampled for comparison (so `texelFetch` survives on ES 3.0), and
//! glsl-in-only modules building at all (the interpolator). The
//! `block.member.length()` re-rooting of runtime-sized storage arrays has
//! its own regression test in horizon-shaders (`translate`), which every
//! change to the pipeline must keep green.

use horizon_shader_api::check::MAX_SAMPLED_TEXTURES_PER_STAGE;
use horizon_shader_api::package::{
    BUILTIN_SHADERS, HORIZON_SHADER_STORE_LIBRARY, builtin_shader_pack,
};
use horizon_shader_api::{Pack, check};
use std::path::Path;

/// Every builtin pack, built as the game ships it: the game writes its
/// builtins to disk as real files and reads them back verbatim, so what
/// ships must pass as it is.
fn builtins() -> Vec<(&'static str, Pack)> {
    BUILTIN_SHADERS
        .iter()
        .chain(HORIZON_SHADER_STORE_LIBRARY.iter())
        .map(|(id, ..)| *id)
        .filter_map(|id| builtin_shader_pack(id).map(|pack| (id, pack)))
        .collect()
}

/// Every stage of a report must come out as GLSL ES 3.0: with the stray
/// builtin-overload types gone from the feature scan, a pack that samples
/// plain textures lands there, which is what a GLES 3.0 phone runs.
fn assert_gles_30(report: &horizon_shader_api::check::Report, id: &str) {
    for stage in &report.stages {
        assert_eq!(
            stage.es_version,
            300,
            "{id}: {} needs GLSL ES {}, which a GLES 3.0 phone cannot run",
            stage.file,
            f32::from(stage.es_version) / 100.0
        );
    }
}

/// The builtin packs (the terrain and post ones the renderer draws with) and
/// the two on disk must pass the check and come out as GLSL ES 3.0 within
/// the binding budget: the check refuses a stage binding over
/// [`MAX_SAMPLED_TEXTURES_PER_STAGE`] sampled textures with a message that
/// says what to do about it, not a bare validation error.
#[test]
fn builtin_and_repo_packs_translate_to_glsl_es_3_0_within_the_binding_budget() {
    let mut packs = builtins();
    for dir in ["packs/golden_hour", "packs/golden_hour_0_2"] {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(dir);
        packs.push((
            dir,
            Pack::read(&path).unwrap_or_else(|e| panic!("{dir}: {e}")),
        ));
    }
    for (id, pack) in &packs {
        let report = check(pack).unwrap_or_else(|problems| {
            panic!("{id} does not pass the check:\n{problems:#?}");
        });
        assert_gles_30(&report, id);
        for stage in &report.stages {
            assert!(
                stage.texture_reads <= 64,
                "{id}: {} reads {} textures",
                stage.file,
                stage.texture_reads
            );
        }
    }
    // The budget the renderer draws against is the GLES one, whatever the
    // device reports over it.
    assert_eq!(MAX_SAMPLED_TEXTURES_PER_STAGE, 16);
}

/// A valid 1x1 RGBA PNG, for the [[textures]] a heavy pack ships: eight of
/// them beside the post stage's prelude is what the fragment stage cannot
/// bind on a GLES phone.
const ONE_BY_ONE_PNG: &[u8] = &[
    137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 73, 72, 68, 82, 0, 0, 0, 1, 0, 0, 0, 1, 8,
    6, 0, 0, 0, 31, 21, 196, 137, 0, 0, 0, 13, 73, 68, 65, 84, 120, 156, 99, 248, 223, 224,
    240, 31, 0, 7, 0, 2, 191, 43, 215, 199, 226, 0, 0, 0, 0, 73, 69, 78, 68, 174, 66, 96, 130,
];

/// A pack that binds more sampled textures than a stage may is refused with
/// a message that says so — the terrain pipeline layout broke on 16-texture
/// GLES phones with a validation error nothing pointed at. Built as a post
/// stage with every prelude texture, its own output target and eight
/// [[textures]]: seventeen in the fragment stage.
#[test]
fn a_pack_over_the_binding_budget_is_refused_with_a_clear_message() {
    // Two passes: the heavy one writes a target of its own, the last draws
    // the picture (a last pass may not have an output of its own).
    let mut manifest = String::from(
        "schema = 1\nid = \"heavy\"\nname = \"Heavy\"\nversion = \"1.0.0\"\napi = \"0.1\"\ndescription = \"Over the GLES budget.\"\nshared = []\n\n[[post]]\nfile = \"post.wgsl\"\nname = \"Heavy pass\"\noutput = \"wide\"\n\n[[post]]\nfile = \"final.wgsl\"\n\n[[options]]\nid = \"glow\"\nname = \"Glow\"\ntype = \"float\"\ndefault = 0.5\nmin = 0.0\nmax = 1.0\n\n",
    );
    let mut files = std::collections::BTreeMap::new();
    for i in 0..8 {
        manifest.push_str(&format!("[[textures]]\nid = \"t{i}\"\nfile = \"t{i}.png\"\n\n"));
        files.insert(format!("t{i}.png"), ONE_BY_ONE_PNG.to_vec());
    }
    manifest.push_str("[budget]\ngpu_ms = 1.0\n");
    files.insert(
        "final.wgsl".into(),
        r#"fn post(input: PostInput) -> vec4<f32> {
    return input.color;
}
"#
        .into(),
    );
    files.insert(
        "post.wgsl".into(),
        r#"override glow: f32 = 0.5;

fn post(input: PostInput) -> vec4<f32> {
    let p = vec2<i32>(input.uv * frame.resolution);
    var acc = vec3<f32>(0.0);
    // Every prelude texture the fragment stage can read, one read each.
    acc += vec3<f32>(textureSampleLevel(scene_color, linear_sampler, input.uv, 0.0).r);
    acc += vec3<f32>(textureLoad(scene_depth, p, 0));
    acc += vec3<f32>(textureSampleCompareLevel(shadow_map, shadow_sampler, input.uv, 0.5));
    acc += vec3<f32>(textureLoad(history, p, 0).r);
    acc += vec3<f32>(textureLoad(shadow_map_translucent, p, 0));
    acc += vec3<f32>(textureSampleLevel(shadow_color, linear_sampler, input.uv, 0.0).r);
    acc += vec3<f32>(bitcast<vec4<f32>>(textureLoad(scene_surface, p, 0)).r);
    // The pass's own output target.
    acc += vec3<f32>(textureSampleLevel(wide, linear_sampler, input.uv, 0.0).r);
    // And the pack's eight.
    acc += vec3<f32>(textureSampleLevel(t0, t0_sampler, input.uv, 0.0).r);
    acc += vec3<f32>(textureSampleLevel(t1, t1_sampler, input.uv, 0.0).r);
    acc += vec3<f32>(textureSampleLevel(t2, t2_sampler, input.uv, 0.0).r);
    acc += vec3<f32>(textureSampleLevel(t3, t3_sampler, input.uv, 0.0).r);
    acc += vec3<f32>(textureSampleLevel(t4, t4_sampler, input.uv, 0.0).r);
    acc += vec3<f32>(textureSampleLevel(t5, t5_sampler, input.uv, 0.0).r);
    acc += vec3<f32>(textureSampleLevel(t6, t6_sampler, input.uv, 0.0).r);
    acc += vec3<f32>(textureSampleLevel(t7, t7_sampler, input.uv, 0.0).r);
    return vec4<f32>(input.color.rgb + acc * glow * 1e-3, input.color.a);
}
"#
        .into(),
    );
    files.insert(horizon_shader_api::MANIFEST.to_owned(), manifest.into_bytes());
    let pack = Pack::from_files(files).expect("the heavy pack is well formed");
    let problems = check(&pack).expect_err("the fragment stage is over the budget");
    let message = problems
        .iter()
        .find(|p| p.message.contains("binds"))
        .unwrap_or_else(|| panic!("no binding message in {problems:#?}"));
    assert!(
        message.message.contains("sampled textures"),
        "{message:#?}"
    );
    assert!(
        message.message.contains("GLES allows 16 per stage"),
        "{message:#?}"
    );
    assert!(
        message
            .message
            .contains("merge small textures into an atlas or a texture array"),
        "{message:#?}"
    );
}
