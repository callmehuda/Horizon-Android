//! Legacy GLES lowering. Vulkan retains the original WGSL byte for byte.
use std::borrow::Cow;

pub(super) fn terrain_shader(gl: bool, arrays: u32) -> Cow<'static, str> {
    // Only the arrays that hold layers are sampled (see wgsl_prefix), so the
    // shader, the layout and the bind group agree on the live count.
    let source = super::wgsl_prefix::sprite_terrain(include_str!("../terrain.wgsl"), arrays);
    if gl {
        Cow::Owned(gles_source(&source, arrays))
    } else {
        Cow::Owned(source)
    }
}

/// GLES 3.0 cannot declare arrays of arrays. Two adjacent vec4 entries
/// retain the original uniform byte layout and every colored-light value.
pub(super) fn gles_uniforms(source: &str) -> String {
    source
        .replace("array<array<vec4<f32>, 2>, 16>", "array<vec4<f32>, 32>")
        .replace(
            "globals.colored_lights[i][0]",
            "globals.colored_lights[2u * i]",
        )
        .replace(
            "globals.colored_lights[i][1]",
            "globals.colored_lights[2u * i + 1u]",
        )
}

fn gles_source(source: &str, arrays: u32) -> String {
    let source = gles_uniforms(source).replace("\r\n", "\n");
    let source = unroll_emissive_lookup(source);
    let start = source
        .find("fn sprite_grad(")
        .expect("terrain sprite function");
    let level = source
        .find("fn sprite_level(")
        .expect("terrain level function");
    let end = level + source[level..].find("\n}\n").expect("terrain level end") + 3;
    let rgss = source
        .find("fn sample_rgss(")
        .expect("terrain RGSS function");
    let rgss_end = rgss + source[rgss..].find("\n}\n").expect("terrain RGSS end") + 3;
    let gles = include_str!("terrain_gles.wgsl.inc")
        .replace(
            "HZ_GLES_SPRITE_GRAD_ARMS",
            &gles_sprite_arms("textureSampleGrad", ", dx, dy", arrays),
        )
        .replace(
            "HZ_GLES_SPRITE_LEVEL_ARMS",
            &gles_sprite_arms("textureSampleLevel", ", lod", arrays),
        );
    let mut output = String::with_capacity(source.len());
    output.push_str(&source[..start]);
    output.push_str(&gles);
    output.push_str(&source[end..rgss]);
    output.push_str(&source[rgss_end..]);

    // Adreno 304 renders broken terrain with the separated scalar varyings.
    // Pack values sharing their interpolation qualifier; leave the vertex
    // buffers, texture sampling, fade, lighting, fog and HDR arithmetic alone.
    let start = output
        .find("struct TerrainOut {")
        .expect("terrain interface");
    let end = output[start..]
        .find("struct SkyOut {")
        .expect("sky interface")
        + start;
    let fields_end = output[start..]
        .find("\n}\n")
        .expect("terrain interface end")
        + start
        + 3;
    let mut body = output[fields_end..end].to_owned();
    for (member, packed) in [
        ("emissive_layer", "codes.w"),
        ("layer", "codes.x"),
        ("emission", "codes.y"),
        ("fade", "codes.z"),
        ("spherical", "distances.x"),
        ("cylindrical", "distances.y"),
        ("visibility", "distances.z"),
        ("height", "distances.w"),
    ] {
        for variable in ["in", "out"] {
            body = body.replace(
                &format!("{variable}.{member}"),
                &format!("{variable}.{packed}"),
            );
        }
    }
    output.replace_range(start..end, &(PACKED_INTERFACE.to_owned() + &body));
    output
}

/// The GLES sprite selection arms for `arrays` live arrays: an if-chain that
/// names only the arrays the pipeline layout binds. A body that so much as
/// mentions an unbound array makes the stage require its binding however
/// dead the branch around it looks — naga does not fold the runtime
/// `SPRITE_ARRAYS == 1u` guard an earlier `terrain_gles.wgsl.inc` carried,
/// so every world with fewer than eight arrays crashed `fs_solid` at
/// pipeline creation on a GLES phone (PowerVR GE8322, one array).
fn gles_sprite_arms(call: &str, tail: &str, arrays: u32) -> String {
    let sample = |name: &str| format!("    return {call}({name}, sprite_sampler, uv, l{tail});\n");
    let name = |i: u32| {
        if i == 0 {
            "sprites".to_owned()
        } else {
            format!("sprites_{i}")
        }
    };
    if arrays <= 1 {
        return sample("sprites");
    }
    let mut out = String::new();
    for i in 0..arrays - 1 {
        out += &format!(
            "    if array_index == {i}u {{\n{}    }}\n",
            sample(&name(i))
        );
    }
    // The last live array takes every index at or past it, which covers the
    // out-of-range ones a corrupt layer word could name as well.
    out += &sample(&name(arrays - 1));
    out
}

// Adreno 710's compiler fails to link this dynamic uniform-array search in a
// robust EGL context. Preserve the first match, every entry and the no-match
// sentinel; constant-index branches also keep robust access enabled.
fn unroll_emissive_lookup(mut source: String) -> String {
    let start = source
        .find("    for (var i = 0u; i < min(globals.emissive_count.x, 32u); i++) {")
        .expect("terrain emissive lookup");
    let body = source[start..].find('{').unwrap() + start;
    let mut depth = 1;
    let mut end = body + 1;
    while depth != 0 {
        match source.as_bytes()[end] {
            b'{' => depth += 1,
            b'}' => depth -= 1,
            _ => {}
        }
        end += 1;
    }
    let mut branches = String::new();
    for i in 0..32 {
        let branch = if i == 0 { "if" } else { "else if" };
        branches.push_str(&format!(
            "    {branch} globals.emissive_count.x > {i}u && globals.emissive_layers[{i}].x == (layer & 0x7fffu) {{\n        out.emissive_layer = globals.emissive_layers[{i}].y;\n    }}\n"
        ));
    }
    source.replace_range(start..end, &branches);
    source
}

const PACKED_INTERFACE: &str = "struct TerrainOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) codes: vec4<u32>,
    @location(2) color: vec4<f32>,
    @location(3) distances: vec4<f32>,
    @location(4) local: vec3<f32>,
}
";

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gfx::wgsl_prefix::sprite_terrain;
    use std::collections::HashSet;

    /// The resource bindings the module's entry points reach — the same
    /// requirement wgpu matches against the pipeline layout: a global an
    /// entry point's call graph references is required however dead the
    /// branch around it looks, which is how the GLES lowering's hardcoded
    /// eight-array if-chain crashed one-array worlds at startup.
    fn entry_point_bindings(module: &naga::Module) -> Vec<naga::ResourceBinding> {
        fn collect(
            module: &naga::Module,
            function: &naga::Function,
            globals: &mut Vec<naga::ResourceBinding>,
            seen: &mut HashSet<naga::Handle<naga::Function>>,
        ) {
            for (_, expression) in function.expressions.iter() {
                if let naga::Expression::GlobalVariable(handle) = *expression {
                    if let Some(binding) = module.global_variables[handle].binding {
                        if !globals.contains(&binding) {
                            globals.push(binding);
                        }
                    }
                }
            }
            walk(module, &function.body, globals, seen);
        }
        fn walk(
            module: &naga::Module,
            block: &naga::Block,
            globals: &mut Vec<naga::ResourceBinding>,
            seen: &mut HashSet<naga::Handle<naga::Function>>,
        ) {
            for statement in block {
                match *statement {
                    naga::Statement::Block(ref inner) => {
                        walk(module, inner, globals, seen);
                    }
                    naga::Statement::If {
                        ref accept,
                        ref reject,
                        ..
                    } => {
                        walk(module, accept, globals, seen);
                        walk(module, reject, globals, seen);
                    }
                    naga::Statement::Switch { ref cases, .. } => {
                        for case in cases {
                            walk(module, &case.body, globals, seen);
                        }
                    }
                    naga::Statement::Loop {
                        ref body,
                        ref continuing,
                        ..
                    } => {
                        walk(module, body, globals, seen);
                        walk(module, continuing, globals, seen);
                    }
                    naga::Statement::Call { function, .. } => {
                        if seen.insert(function) {
                            collect(module, &module.functions[function], globals, seen);
                        }
                    }
                    _ => {}
                }
            }
        }
        let mut globals = Vec::new();
        let mut seen = HashSet::new();
        for entry in &module.entry_points {
            collect(module, &entry.function, &mut globals, &mut seen);
        }
        globals
    }

    /// The bindings the renderer's terrain layout holds for `arrays` arrays
    /// (gfx.rs): the globals, the live sprite arrays and their sampler, the
    /// lightmap and its sampler, and the emission thresholds. `sprites_1..`
    /// only exist from the second array on.
    fn live_bindings(arrays: u32) -> Vec<u32> {
        [0, 1, 2, 3, 4, 12]
            .into_iter()
            .chain(5..5 + arrays.saturating_sub(1))
            .collect()
    }

    /// The flat draws' group 1 (terrain.wgsl's `block_slots` and
    /// `slot_origins`): storage, not sampled textures, on its own layout.
    const FLAT_DRAW_BINDINGS: &[u32] = &[0, 1];

    #[test]
    fn vulkan_source_is_unchanged_and_gles_validates_every_entrypoint() {
        // Eight arrays: every arm the file can hold, one fewer mention of
        // any array with one (the layout, the bind group and the shader agree
        // on the live count; see wgsl_prefix).
        let vulkan = terrain_shader(false, 8);
        assert_eq!(vulkan, sprite_terrain(include_str!("../terrain.wgsl"), 8));
        assert!(vulkan.contains("sprites_7"));
        let one = terrain_shader(false, 1);
        assert!(!one.contains("textureSampleGrad(sprites_1"));
        let source = terrain_shader(true, 8);
        let original = sprite_terrain(include_str!("../terrain.wgsl"), 8).replace("\r\n", "\n");
        assert_eq!(
            &source[source.find("struct SkyOut {").unwrap()..],
            &original[original.find("struct SkyOut {").unwrap()..],
            "sky and outline shader bodies must remain unchanged"
        );
        assert!(!source.contains("array<array<"));
        for arrays in [1u32, 2, 3, 8] {
            let source = terrain_shader(true, arrays);
            let module = naga::front::wgsl::parse_str(&source).unwrap();
            naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::default(),
            )
            .validate(&module)
            .unwrap();
        }
    }

    /// Every array count the renderer can specialize to: the GLES module's
    /// entry points must require exactly the bindings the layout holds —
    /// a stage that so much as mentions `sprites_1` from a one-array world
    /// is refused (`ResourceBinding { group: 0, binding: 5 } is not
    /// available in the pipeline layout`), which crashed the terrain
    /// pipeline at startup on GLES phones (PowerVR GE8322).
    #[test]
    fn gles_entry_points_require_only_the_live_sprite_bindings() {
        for arrays in 1..=8u32 {
            let source = terrain_shader(true, arrays);
            let module = naga::front::wgsl::parse_str(&source).unwrap();
            naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::default(),
            )
            .validate(&module)
            .unwrap();
            let live = live_bindings(arrays);
            for binding in entry_point_bindings(&module) {
                // Group 1 is the flat draws' `block_slots`/`slot_origins`, a
                // layout of its own (flat.rs); the terrain arrays are group 0.
                let allowed = match binding.group {
                    0 => live.as_slice(),
                    1 => FLAT_DRAW_BINDINGS,
                    group => panic!(
                        "{arrays} arrays: the terrain module draws from group {group}, which no terrain layout holds"
                    ),
                };
                assert!(
                    allowed.contains(&binding.binding),
                    "{arrays} arrays: a stage requires binding {} in group {}, which the \
                     layout for {} holds only {:?}",
                    binding.binding,
                    binding.group,
                    if binding.group == 0 {
                        "its live sprite arrays"
                    } else {
                        "the flat section draws"
                    },
                    allowed
                );
            }
            // The arms name no array past the live count, either: the
            // declarations stay (harmless when unused), the samples must not.
            for beyond in arrays..8 {
                assert!(
                    !source.contains(&format!("(sprites_{beyond}")),
                    "{arrays} arrays: the GLES arms still sample sprites_{beyond}"
                );
            }
        }
    }
}
