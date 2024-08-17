# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Assets module for USD exporter."""
import os
from typing import List

import mujoco
from PIL import Image as im
from PIL import ImageOps

class Texture:
  """Wrapper class for textures defined in mjModel."""

  def __init__(
      self,
      texid: int,
      model: mujoco.MjModel,
      frames_directory: str,
      assets_directory: str,
      shareable: bool
  ) -> None:
    self.texid = texid
    self.model = model
    self.frames_directory = frames_directory
    self.assets_directory = assets_directory
    self.shareable = shareable

    self.create_texture_asset_file()

  @property
  def path(self) -> str:
    return self.img_path

  @property
  def type(self) -> mujoco.mjtTexture:
    return self.model.tex_type[self.texid]

  def create_texture_asset_file(self):
    tex_adr = self.model.tex_adr[self.texid]
    tex_height = self.model.tex_height[self.texid]
    tex_width = self.model.tex_width[self.texid]
    tex_nchannel = self.model.tex_nchannel[self.texid]
    pixels = tex_nchannel * tex_height * tex_width
    img = im.fromarray(
        self.model.tex_data[tex_adr : tex_adr + pixels].reshape(
            tex_height, tex_width, 3
        )
    )
    img = ImageOps.flip(img)

    texture_file_name = f"texture_{self.texid}.png"

    # saves texture asset to the specified assets directory
    texture_path = os.path.join(self.assets_directory, texture_file_name)
    img.save(texture_path)

    relative_path = os.path.relpath(
        self.assets_directory, self.frames_directory
    )
    
    if self.shareable:
      self.img_path = os.path.join(
        relative_path, texture_file_name
    )
    else:
      self.img_path = os.path.abspath(texture_path)

class Material:
  """Wrapper class for materials defined in mjModel."""

  def __init__(
      self,
      matid: int,
      model: mujoco.MjModel,
      textures: List[Texture],
  ) -> None:
    self.matid = matid
    self.model = model
    self.textures = textures

  @property
  def texuniform(self) -> bool:
    return self.model.mat_texuniform[self.matid]

  @property
  def texrepeat(self) -> List[float]:
    return self.model.mat_texrepeat[self.matid]

  @property
  def emission(self) -> float:
    return self.model.mat_emission[self.matid]

  @property
  def specular(self) -> float:
    return self.model.mat_specular[self.matid]

  @property
  def shininess(self) -> float:
    return self.model.mat_shininess[self.matid]

  @property
  def reflectance(self) -> float:
    return self.model.mat_reflectance[self.matid]

  @property
  def metallic(self) -> float:
    return self.model.mat_metallic[self.matid]

  @property
  def roughness(self) -> float:
    return self.model.mat_roughness[self.matid]

  @property
  def rgba(self) -> List[float]:
    return self.model.mat_rgba[self.matid]

  @property
  def rgb_tex(self) -> Texture:
    return self.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB]

  @property
  def occlusion_tex(self) -> Texture:
    return self.textures[mujoco.mjtTextureRole.mjTEXROLE_OCCLUSION]

  @property
  def roughness_tex(self) -> Texture:
    return self.textures[mujoco.mjtTextureRole.mjTEXROLE_ROUGHNESS]

  @property
  def metallic_tex(self) -> Texture:
    return self.textures[mujoco.mjtTextureRole.mjTEXROLE_METALLIC]

  @property
  def normal_tex(self) -> Texture:
    return self.textures[mujoco.mjtTextureRole.mjTEXROLE_NORMAL]

  @property
  def opacity_tex(self) -> Texture:
    return self.textures[mujoco.mjtTextureRole.mjTEXROLE_OPACITY]

  @property
  def emissive_tex(self) -> Texture:
    return self.textures[mujoco.mjtTextureRole.mjTEXROLE_EMISSIVE]

  @property
  def rgba_tex(self) -> Texture:
    return self.textures[mujoco.mjtTextureRole.mjTEXROLE_RGBA]

  @property
  def orm_tex(self) -> Texture:
    return self.textures[mujoco.mjtTextureRole.mjTEXROLE_ORM]
