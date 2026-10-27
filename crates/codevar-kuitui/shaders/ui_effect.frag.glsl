#version 450

// Layer-effect fragment shader.
//
// One evaluation of the signed distance of a rounded box gives the
// rasterizer every effect a layer asks for, which is why no effect
// needs a texture pass:
//
// * blur is the Gaussian integral of the box, whose closed form along
//   the distance axis is an erfc, so the cost does not grow with the
//   radius the way an iterated convolution would;
// * rotation maps the sample point into the frame of the shape instead
//   of moving the vertices, so the geometry of the layer is shared with
//   the plain UI program;
// * pixelate quantizes the sample point of the fill, which turns the
//   ramp into the mosaic of its own colors;
// * gradient evaluates the ramp between the two endpoints of the axis;
// * stroke subtracts the box inset by its own width from the box, which
//   is how an outlined rectangle follows its own corners instead of
//   being drawn as four straight edges.
//
// The corner radius is chosen per quadrant, which is exact as long as
// the radii do not overlap along an edge — the same bound the layout
// clamps a shape to.

layout(location = 0) in vec4 v_color;
layout(location = 1) in vec2 v_position;

layout(push_constant) uniform PushConstants {
    vec4 color_from;
    vec4 color_to;
    vec2 viewport;
    vec2 origin;
    vec2 size;
    vec2 axis_start;
    vec2 axis_end;
    vec2 blur_pixel;
    vec4 radii;
    float rotation;
    uint kinds;
    float stroke;
} pc;

layout(location = 0) out vec4 out_color;

const uint KIND_BLUR = 1u;
const uint KIND_PIXELATE = 2u;
const uint KIND_ROTATE = 4u;
const uint KIND_GRADIENT = 8u;
const uint KIND_STROKE = 16u;

/// The distance to the rounded box of half-size `b` and corner radius
/// `r`, negative inside.
float sd_round_box(vec2 p, vec2 b, float r) {
    vec2 q = abs(p) - b + r;
    return min(max(q.x, q.y), 0.0) + length(max(q, vec2(0.0))) - r;
}

/// The rotation of `p` by `angle`.
vec2 rotate(vec2 p, float angle) {
    float c = cos(angle);
    float s = sin(angle);
    return vec2(c * p.x - s * p.y, s * p.x + c * p.y);
}

/// The complementary error function, from the Abramowitz & Stegun 7.1.26
/// rational approximation (maximum error 1.5e-7, far below one eighth of
/// a tone step). GLSL has no `erfc`, and the approximation is the usual
/// way a shader reaches a Gaussian.
float erfc_approx(float x) {
    float magnitude = abs(x);
    float t = 1.0 / (1.0 + 0.3275911 * magnitude);
    float polynomial = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741
        + t * (-1.453152027 + t * 1.061405429))));
    float positive = 1.0 - polynomial * exp(-magnitude * magnitude);
    // erfc is symmetric about (0, 1), which the sign test restores without
    // letting the rational form see a negative argument.
    return x < 0.0 ? 2.0 - positive : positive;
}

/// The coverage the edge at signed distance `dist` asks for.
///
/// With a positive `sigma` the edge is the analytic falloff of a Gaussian of
/// that standard deviation, whose cost does not grow with the radius; without
/// one, the edge is the antialiased step of the box.
float edge_coverage(float dist, float sigma) {
    if (sigma > 0.0) {
        return clamp(0.5 * erfc_approx(dist / sigma), 0.0, 1.0);
    }
    float aa = max(fwidth(dist), 1e-5);
    return clamp(0.5 - dist / aa, 0.0, 1.0);
}

void main() {
    vec2 center = pc.origin + pc.size * 0.5;
    vec2 half_size = pc.size * 0.5;
    // Sample in the frame of the shape, so a rotated layer needs no
    // rotated geometry.
    vec2 p = v_position - center;
    if ((pc.kinds & KIND_ROTATE) != 0u) {
        p = rotate(p, -pc.rotation);
    }
    // The radius of the corner of the quadrant being sampled: top-left
    // and top-right for a negative y, bottom-left and bottom-right for a
    // positive one, because the UI origin is at the top-left.
    float top = mix(pc.radii.x, pc.radii.y, step(0.0, p.x));
    float bottom = mix(pc.radii.w, pc.radii.z, step(0.0, p.x));
    float radius = mix(top, bottom, step(0.0, p.y));
    float dist = sd_round_box(p, half_size, radius);

    float sigma = 0.0;
    if ((pc.kinds & KIND_BLUR) != 0u && pc.blur_pixel.x > 0.0) {
        sigma = pc.blur_pixel.x * 1.4142135;
    }
    float coverage = edge_coverage(dist, sigma);
    if ((pc.kinds & KIND_STROKE) != 0u && pc.stroke > 0.0) {
        // The outline is the box minus the box inset by its own width, so
        // it follows whatever corners the layer already carries. The width
        // is clamped to half the smaller axis the same way the geometry
        // expansion clamps a stroke, so a band never inverts the inner box.
        float width = min(pc.stroke, min(half_size.x, half_size.y));
        float inner = sd_round_box(p, half_size - vec2(width), max(radius - width, 0.0));
        coverage = clamp(coverage - edge_coverage(inner, sigma), 0.0, 1.0);
    }

    vec2 fill = p;
    if ((pc.kinds & KIND_PIXELATE) != 0u && pc.blur_pixel.y > 0.0) {
        float grid = pc.blur_pixel.y;
        fill = floor(p / grid) * grid + grid * 0.5;
    }
    vec4 color = v_color;
    if ((pc.kinds & KIND_GRADIENT) != 0u) {
        vec2 point = center + fill;
        if ((pc.kinds & KIND_ROTATE) != 0u) {
            point = center + rotate(fill, pc.rotation);
        }
        vec2 axis = pc.axis_end - pc.axis_start;
        float length_squared = dot(axis, axis);
        float factor = 0.0;
        if (length_squared > 0.0) {
            factor = dot(point - pc.axis_start, axis) / length_squared;
        }
        color = mix(pc.color_from, pc.color_to, clamp(factor, 0.0, 1.0));
    }

    float alpha = color.a * coverage;
    if (alpha <= 0.003921569) {
        discard;
    }
    out_color = vec4(color.rgb, alpha);
}
