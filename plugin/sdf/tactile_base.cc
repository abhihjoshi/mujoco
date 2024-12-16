#include <cstdint>
#include <optional>
#include <utility>

#include <mujoco/mjplugin.h>
#include <mujoco/mjtnum.h>
#include <mujoco/mujoco.h>
#include "sdf.h"
#include "tactile_base.h"

namespace mujoco::plugin::sdf {
namespace {

static mjtNum distance(const mjtNum p[3], const mjtNum gap[1]) {
    return mju_sqrt(p[0]*p[0] + p[1]*p[1] + p[2]*p[2]) - gap[0];
}

}  // namespace

// factory function
std::optional<TactileBase> TactileBase::Create(
    const mjModel* m, mjData* d, int instance) {
  if (CheckAttr("gap", m, instance)) {
    return TactileBase(m, d, instance);
  } else {
    mju_warning("Invalid gap parameter for TactileBase plugin");
    return std::nullopt;
  }
}

// plugin constructor
TactileBase::TactileBase(const mjModel* m, mjData* d, int instance) {
  SdfDefault<TactileBaseAttribute> defattribute;

  for (int i=0; i < TactileBaseAttribute::nattribute; i++) {
    attribute[i] = defattribute.GetDefault(
        TactileBaseAttribute::names[i],
        mj_getPluginConfig(m, instance, TactileBaseAttribute::names[i]));
  }
}

// sdf
mjtNum TactileBase::Distance(const mjtNum point[3]) const {
  return distance(point, attribute);
}

// gradient of sdf
void TactileBase::Gradient(mjtNum grad[3], const mjtNum point[3]) const {
  mjtNum eps = 1e-8;
  mjtNum dist0 = distance(point, attribute);
  mjtNum pointX[3] = {point[0]+eps, point[1], point[2]};
  mjtNum distX = distance(pointX, attribute);
  mjtNum pointY[3] = {point[0], point[1]+eps, point[2]};
  mjtNum distY = distance(pointY, attribute);
  mjtNum pointZ[3] = {point[0], point[1], point[2]+eps};
  mjtNum distZ = distance(pointZ, attribute);

  grad[0] = (distX - dist0) / eps;
  grad[1] = (distY - dist0) / eps;
  grad[2] = (distZ - dist0) / eps;
}

// plugin registration
void TactileBase::RegisterPlugin() {
  mjpPlugin plugin;
  mjp_defaultPlugin(&plugin);

  plugin.name = "mujoco.sdf.tactile_base";
  plugin.capabilityflags |= mjPLUGIN_SDF;

  plugin.nattribute = TactileBaseAttribute::nattribute;
  plugin.attributes = TactileBaseAttribute::names;
  plugin.nstate = +[](const mjModel* m, int instance) { return 0; };

  plugin.init = +[](const mjModel* m, mjData* d, int instance) {
    auto sdf_or_null = TactileBase::Create(m, d, instance);
    if (!sdf_or_null.has_value()) {
      return -1;
    }
    d->plugin_data[instance] = reinterpret_cast<uintptr_t>(
        new TactileBase(std::move(*sdf_or_null)));
    return 0;
  };
  plugin.destroy = +[](mjData* d, int instance) {
    delete reinterpret_cast<TactileBase*>(d->plugin_data[instance]);
    d->plugin_data[instance] = 0;
  };
  plugin.reset = +[](const mjModel* m, mjtNum* plugin_state, void* plugin_data,
                     int instance) {
    // do nothing
  };
  plugin.compute =
      +[](const mjModel* m, mjData* d, int instance, int capability_bit) {
        // do nothing;
      };
  plugin.sdf_distance =
      +[](const mjtNum point[3], const mjData* d, int instance) {
        auto* sdf = reinterpret_cast<TactileBase*>(d->plugin_data[instance]);
        return sdf->Distance(point);
      };
  plugin.sdf_gradient = +[](mjtNum gradient[3], const mjtNum point[3],
                        const mjData* d, int instance) {
    auto* sdf = reinterpret_cast<TactileBase*>(d->plugin_data[instance]);
    sdf->Gradient(gradient, point);
  };
  plugin.sdf_staticdistance =
      +[](const mjtNum point[3], const mjtNum* attributes) {
        return distance(point, attributes);
      };
  plugin.sdf_aabb =
      +[](mjtNum aabb[6], const mjtNum* attributes) {
        aabb[0] = aabb[1] = aabb[2] = 0;
        aabb[3] = aabb[4] = aabb[5] = attributes[0];
      };
  plugin.sdf_attribute =
      +[](mjtNum attribute[], const char* name[], const char* value[]) {
        SdfDefault<TactileBaseAttribute> defattribute;
        defattribute.GetDefaults(attribute, name, value);
      };

  mjp_registerPlugin(&plugin);
}

}  // namespace mujoco::plugin::sdf