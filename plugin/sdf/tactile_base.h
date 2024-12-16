#ifndef MUJOCO_PLUGIN_SDF_TACTILE_BASE_H_
#define MUJOCO_PLUGIN_SDF_TACTILE_BASE_H_

#include <optional>

#include <mujoco/mjdata.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mjtnum.h>
#include <mujoco/mjvisualize.h>
#include "sdf.h"

namespace mujoco::plugin::sdf {

struct TactileBaseAttribute {
  static constexpr int nattribute = 1;
  static constexpr char const* names[nattribute] = {"gap"};
  static constexpr mjtNum defaults[nattribute] = {0.001};
};

class TactileBase {
 public:
  // Creates a new TactileBase instance (allocated with `new`) or
  // returns null on failure.
  static std::optional<TactileBase> Create(const mjModel* m, mjData* d, int instance);
  TactileBase(TactileBase&&) = default;
  ~TactileBase() = default;

  void Reset();
  void Visualize(const mjModel* m, mjData* d, const mjvOption* opt,
                 mjvScene* scn, int instance);
  void Compute(const mjModel* m, mjData* d, int instance);
  mjtNum Distance(const mjtNum point[3]) const;
  void Gradient(mjtNum grad[3], const mjtNum point[3]) const;

  static void RegisterPlugin();

  mjtNum attribute[TactileBaseAttribute::nattribute];

 private:
  TactileBase(const mjModel* m, mjData* d, int instance);

  SdfVisualizer visualizer_;
};

}  // namespace mujoco::plugin::sdf

#endif
