@group(0) @binding(0) var<uniform> uniforms : Uniforms;
@group(0) @binding(1) var<storage> jitter: array<vec2f>; 
@group(0) @binding(2) var<storage> positions: array<vec3f>; 
@group(0) @binding(3) var<storage> indices: array<vec3u>;
@group(0) @binding(4) var<storage> normals: array<vec3f>;
@group(0) @binding(5) var<uniform> aabb: Aabb;
@group(0) @binding(6) var<storage> treeIds: array<u32>;
@group(0) @binding(7) var<storage> bspTree: array<vec4u>;
@group(0) @binding(8) var<storage> bspPlanes: array<f32>;

const Pi = 3.141592; // Pi, but not
const lightPosition = vec3f(0.0, 1.0, 0.0);
const epsilon = 0.0001;
const MAX_LEVEL = 20u; 
const BSP_LEAF = 3u;
var  <private> branch_node: array<vec2u, MAX_LEVEL>;
var  <private> branch_ray: array<vec2f, MAX_LEVEL>;

struct Uniforms {
    aspect: f32,
    camera_constant: f32,
    jitterSub: f32,
};

struct Aabb
{
    min: vec3f,
    max: vec3f,
};

struct VSOut
{
    @builtin(position) position: vec4f,
    @location(0) coords : vec2f
};

struct Ray 
{
    origin: vec3f,
    direction: vec3f,
    tmin: f32,
    tmax: f32
};

struct Light
{
    L_i: vec3f,
    direction: vec3f,
    dist: f32
};

struct HitInfo
{
    has_hit: bool,
    dist: f32,
    position: vec3f,
    normal: vec3f,
    diffusecolor: vec3f,
    emission: vec3f,
};

@fragment
fn main_fs(@location(0) coords: vec2f) -> @location(0) vec4f
{
    const bgcolor = vec4f(0.1, 0.3, 0.6, 1.0);
    const max_depth = 10;

    var result = vec3f(0.0);
    for (var j = 0u; j < u32(uniforms.jitterSub*uniforms.jitterSub); j++) 
    {
        let uv = vec2f(coords.x*uniforms.aspect*0.5f, coords.y*0.5f);
        var r = get_camera_ray(uv+jitter[j]);
        var hit = HitInfo(false, 0.0, vec3f(0.0), vec3f(0.0), vec3f(0.0), vec3f(0.0));

        for(var i = 0; i < max_depth; i++) 
        {
            if(intersect_scene(&r, &hit)) { 
                result += shade(&r, &hit);
            }
            else { 
                result += bgcolor.rgb;
                break; 
            }

            if(hit.has_hit) { break; }
        }
    }
    result /= uniforms.jitterSub*uniforms.jitterSub;
    return vec4f(pow(result, vec3f(0.66)), bgcolor.a);
}

@vertex
fn main_vs(@builtin(vertex_index) VertexIndex : u32) -> VSOut
{
    let unused = normals[0];
    const pos = array<vec2f, 4>(vec2f(-1.0, 1.0), vec2f(-1.0, -1.0), vec2f(1.0, 1.0), vec2f(1.0, -1.0));
    var vsOut: VSOut;
    vsOut.position = vec4f(pos[VertexIndex], 0.0, 1.0);
    vsOut.coords = pos[VertexIndex];
    return vsOut;
}



    
fn get_camera_ray(uv: vec2f) -> Ray
{
    const eye = vec3f(-0.02, 0.11, 0.5);
    const point = vec3(-0.02, 0.11, 0.0);
    const up = vec3f(0.0, 1.0, 0.0);
    var d = uniforms.camera_constant;
    var v = normalize(point-eye);
    var b1 = normalize(cross(v, up));
    var b2 = cross(b1, v);
    var q = b1 * uv.x + b2 * uv.y + v * d;
    var omega = normalize(q);
    
    var ray = Ray(eye, omega, 0.0, 1.0e32);
    return ray;
}


fn sample_point_light(pos: vec3f) -> Light
    {
        const lightIntensity = vec3f(Pi, Pi, Pi); 
        
        var L_i = lightIntensity / dot((lightPosition - pos), (lightPosition - pos));

        var l = lightPosition - pos;

        var dist = length(l);
        
        var light = Light(L_i, normalize(l), dist);
        return light;
    }

fn sample_directional_light() -> Light 
{
    const intensityGAINS = 1;
    const lightIntensity = vec3f(Pi*intensityGAINS, Pi*intensityGAINS, Pi*intensityGAINS);
    var L_i = lightIntensity;
    var l = normalize(vec3f(-1.0, -1.0, -1.0));
    var dist = 1e32;
    var light = Light(L_i, l, dist);
    return light;
}



fn intersect_scene(r: ptr<function, Ray>, hit : ptr<function, HitInfo>) -> bool
{
    var triangleRGB = vec3f(0.9);
    if (intersect_min_max(r))
    {
        if (intersect_trimesh(r, hit)) 
        {
            (*r).tmax = min((*hit).dist, (*r).tmax);
            (*hit).has_hit = true;
            if ((*r).tmax == (*hit).dist) {
                (*hit).diffusecolor = triangleRGB;
            }
        }
    }
    return (*hit).has_hit;
}

fn intersect_triangle(r: Ray, hit: ptr<function, HitInfo>, index: u32) -> bool 
{
    let temp = indices[index]; 
    var e0 = positions[temp.y] - positions[temp.x];
    var e1 = positions[temp.z] - positions[temp.x];
    var n = cross(e0, e1);
    
    var dotMan = dot(r.direction, n);
    if (abs(dotMan) < 1e-8)
    {return false;}
    var t = (dot((positions[temp.x] - r.origin), n)) / dotMan;
    
    var objectHit = false;
    if ((t < r.tmax) & (t > r.tmin)) {
        var beta = dot(cross((positions[temp.x]-r.origin), r.direction), e1)/dot(r.direction, n);
        var gamma = -dot(cross((positions[temp.x]-r.origin), r.direction), e0)/dot(r.direction, n);
        var alpha = 1 - beta - gamma;
        objectHit = (beta >= 0) & (gamma >= 0) & ((beta + gamma) <= 1);

        if (objectHit) {
            if ((*hit).dist > 0) {
                (*hit).dist = min((*hit).dist, t);
            }
            else {
                (*hit).dist = t;
            }
            (*hit).position = r.origin + t * r.direction;
            var x = alpha * normals[temp.x] + beta * normals[temp.y] + gamma * normals[temp.z];
            (*hit).normal = normalize(x);
        }
    }
    
    return objectHit;
    
}

fn shade(r: ptr<function, Ray>, hit: ptr<function, HitInfo>) -> vec3f
    {   
        var light = sample_directional_light();

        var ray = Ray((*hit).position, -light.direction, epsilon, light.dist-epsilon);
        var temp = HitInfo(false, 0.0, vec3f(0.0), vec3f(0.0), vec3f(0.0), vec3f(0.0));

        return ((*hit).diffusecolor/Pi) * 1 * light.L_i * dot(-light.direction, (*hit).normal);
    }

fn  intersect_min_max(r: ptr  <function, Ray  >)   -> bool 
{ 
    let   p1 = (aabb.min - r.origin)/r.direction;
    let   p2 = (aabb.max - r.origin)/r.direction;
    let   pmin = min(p1, p2);
    let   pmax = max(p1, p2);
    let   tmin = max(pmin.x, max(pmin.y, pmin.z));
    let   tmax = min(pmax.x, min(pmax.y, pmax.z));
    if (tmin > tmax || tmin > r.tmax || tmax < r.tmin) 
    {
        return false; 
    } 
    r.tmin = max(tmin - 1.0e-3f,   r.tmin);
    r.tmax = min(tmax + 1.0e-3f,   r.tmax);
    return true; 
}
    
fn intersect_trimesh(r: ptr<function, Ray>, hit: ptr<function, HitInfo>) -> bool {
    var branch_lvl = 0u;
    var near_node = 0u;
    var far_node = 0u;
    var t = 0.0f;
    var node = 0u;

    for (var i = 0u; i <= MAX_LEVEL; i++) {
        let tree_node = bspTree[node];
        let node_axis_leaf = tree_node.x & 3u;

        if (node_axis_leaf == BSP_LEAF) {
            // A leaf was found
            let node_count = tree_node.x >> 2u;
            let node_id = tree_node.y;
            var found = false;

            for (var j = 0u; j < node_count; j++) {
                let obj_idx = treeIds[node_id + j];
                if (intersect_triangle(*r, hit, obj_idx)) {
                    r.tmax = hit.dist;
                    found = true;
                }
            }

            if (found) {
                return true;
            } else if (branch_lvl == 0u) {
                return false;
            } else {
                branch_lvl--;
                i = branch_node[branch_lvl].x;
                node = branch_node[branch_lvl].y;
                r.tmin = branch_ray[branch_lvl].x;
                r.tmax = branch_ray[branch_lvl].y;
                continue;
            }

        }

        let axis_direction = r.direction[node_axis_leaf];
        let axis_origin = r.origin[node_axis_leaf];

        if (axis_direction >= 0.0f) {
            near_node = tree_node.z; // left
            far_node = tree_node.w;  // right
        } else {
            near_node = tree_node.w; // right
            far_node = tree_node.z;  // left
        }

        let node_plane = bspPlanes[node];
        let denom = select(axis_direction, 1.0e-8f, abs(axis_direction) < 1.0e-8f);
        t = (node_plane - axis_origin) / denom;

        if (t > r.tmax) {
            node = near_node;
        } else if (t < r.tmin) {
            node = far_node;
        } else {
            branch_node[branch_lvl].x = i;
            branch_node[branch_lvl].y = far_node;
            branch_ray[branch_lvl].x = t;
            branch_ray[branch_lvl].y = r.tmax;
            branch_lvl++;
            r.tmax = t;
            node = near_node;
        }
    }

    return false;
}
