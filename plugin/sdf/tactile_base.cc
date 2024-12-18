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

// checks if the point is within some z bound of the bottom
static bool checkBounds(const mjtNum p[3], mjtNum x, mjtNum y, mjtNum z, bool checkZ) {
  mjtNum p_abs[3] = { abs(p[0]), abs(p[1]), abs(p[2]) };
  return p_abs[0] <= x && p_abs[1] <= y && (!checkZ || p_abs[2] <= z);
}

static mjtNum distanceInserted(const mjtNum p[3], mjtNum x, mjtNum y) {
  mjtNum p_abs[3] = { abs(p[0]), abs(p[1]), abs(p[2]) };
  return mju_min(x - p_abs[0], mju_min(y - p_abs[1], p_abs[2]));
}

static mjtNum distanceEngaged(const mjtNum p[3], mjtNum x, mjtNum y, mjtNum z) {
  mjtNum p_abs[3] = { abs(p[0]), abs(p[1]), abs(p[2]) };
  mjtNum q[3] = { x - p_abs[0], y - p_abs[1], p_abs[2] - z };
  mjtNum xDist = mju_sqrt(q[0] * q[0] + q[2] * q[2]);
  mjtNum yDist = mju_sqrt(q[1] * q[1] + q[2] * q[2]);
  return mju_min(p_abs[2], mju_min(xDist, yDist));
}

static mjtNum distanceTop(const mjtNum p[3], const mjtNum topSize[3]) {

  // distance if inserted
  if (checkBounds(p, topSize[0] - 0.005, topSize[1] - 0.005, topSize[2], true)) {
    return distanceInserted(p, topSize[0] - 0.005, topSize[1] - 0.005);
  }
  // distance if engaged
  else if (checkBounds(p, topSize[0] - 0.005, topSize[1] - 0.005, 0, false)) {
    return distanceEngaged(p, topSize[0] - 0.005, topSize[1] - 0.005, topSize[2]);
  }

  // distance if not inserted or engaged 
  mjtNum q[3] = { abs(p[0]) - topSize[0], abs(p[1]) - topSize[1], abs(p[2]) - topSize[2] };
  mjtNum q_abs[3] = { mju_max(q[0], 0), mju_max(q[1], 0), mju_max(q[2], 0) };
  mjtNum q_len = mju_sqrt(q_abs[0]*q_abs[0] + q_abs[1]*q_abs[1] + q_abs[2]*q_abs[2]);
  mjtNum defaultDistance = q_len + mju_min(mju_max(q[0], mju_max(q[1], q[2])), 0);

  return defaultDistance;
}

static mjtNum distanceBottom(const mjtNum p[3], const mjtNum bottomSize[3]) {
  mjtNum q[3] = { abs(p[0]) - bottomSize[0], abs(p[1]) - bottomSize[1], abs(p[2]) - bottomSize[2] };
  mjtNum q_abs[3] = { mju_max(q[0], 0), mju_max(q[1], 0), mju_max(q[2], 0) };
  mjtNum q_len = mju_sqrt(q_abs[0]*q_abs[0] + q_abs[1]*q_abs[1] + q_abs[2]*q_abs[2]);
  return q_len + mju_min(mju_max(q[0], mju_max(q[1], q[2])), 0);
}

static mjtNum distance(const mjtNum p[3], const mjtNum gap[1]) {
    const mjtNum bottomSize[3] = { 0.05, 0.05, 0.01 }; // half length of the bottom box
    const mjtNum topSize[3] = { (0.035 + 2 * gap[0]) / 2, 
                                (0.035 + 2 * gap[0]) / 2,
                                0.025 };

    // SDF of the bottom of the base
    mjtNum offsetPointBottom[3];
    offsetPointBottom[0] = p[0];
    offsetPointBottom[1] = p[1];
    offsetPointBottom[2] = p[2] - bottomSize[2];

    // SDF of the top of the base
    mjtNum offsetPointTop[3];
    offsetPointTop[0] = p[0];
    offsetPointTop[1] = p[1];
    offsetPointTop[2] = p[2] - (2 * bottomSize[2] + topSize[2]);

    return mju_min(distanceTop(offsetPointTop, topSize), distanceBottom(offsetPointBottom, bottomSize));
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
        aabb[0] = aabb[1] = 0;
        aabb[2] = (0.02 + 0.05) / 2;
        aabb[3] = aabb[4] = 0.05;
        aabb[5] = (0.02 + 0.05) / 2;
      };
  plugin.sdf_attribute =
      +[](mjtNum attribute[], const char* name[], const char* value[]) {
        SdfDefault<TactileBaseAttribute> defattribute;
        defattribute.GetDefaults(attribute, name, value);
      };

  mjp_registerPlugin(&plugin);
}

}  // namespace mujoco::plugin::sdf