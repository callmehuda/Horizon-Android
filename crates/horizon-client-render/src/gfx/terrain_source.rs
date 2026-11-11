//! Legacy GLES lowering. Vulkan retains the original WGSL byte for byte.
use std::borrow::Cow;

pub(super) fn terrain_shader(gl: bool, arrays: u32) -> Cow<'static, str> {
    // Only the arrays that hold layers are sampled (see wgsl_prefix), so the
    // shader, the layout and the bind group agree on the live count.
    let source = super::wgsl_prefix::sprite_terrain(include_str!("../terrain.wgsl"), arrays);
    if gl {
        Cow::Owned(gles_source(&source))
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

fn gles_source(source: &str) -> String {
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
    let mut output = String::with_capacity(source.len());
    output.push_str(&source[..start]);
    output.push_str(include_str!("terrain_gles.wgsl.inc"));
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
    use crate::gfx::wgsl_prefix::sprite_terrain;
    use super::*;

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
}
