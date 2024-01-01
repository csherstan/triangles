import dataclasses
from enum import Enum
from functools import partial
from typing import SupportsFloat, Any, List, Dict, Tuple

import numpy as np
from PIL import Image, ImageDraw
from gymnasium import Env
from gymnasium.core import spaces, RenderFrame

ActType = Dict[str, Dict[str, Any]]
ObsType = spaces.Dict

from ray.rllib.models.tf.tf_modelv2 import TFModelV2
@dataclasses.dataclass
class Point:
  x: float
  y: float


@dataclasses.dataclass(frozen=True)
class Triangle:
  """
  This is meant to be a semantic mapping between and np.ndarray and the components of the triangle definition
  """
  r: float  # [0, 1]
  g: float  # [0, 1]
  b: float  # [0, 1]
  a: float  # [0, 1]

  # units for each axis should be [0, 1]
  vertices: Tuple[Point, Point, Point]
  array: np.ndarray

  @staticmethod
  def from_array(arr: np.ndarray) -> "Triangle":
    r = arr[0]
    g = arr[1]
    b = arr[2]
    a = arr[3]
    vertices = []
    for i in range(3):
      offset = i * 2 + 4
      vertices.append(Point(*arr[offset:offset + 2]))

    return Triangle(r=r, g=g, b=b, a=a, vertices=tuple(vertices), array=arr)


class TriangleEnv(Env[ActType, ObsType]):
  target_image: np.ndarray
  triangles: List[Triangle]
  rendered: np.ndarray
  height: int
  width: int
  add_cost: float
  edit_cost: float
  energy: float
  max_alpha: float

  class Op(Enum):
    STOP = 0
    EDIT = 1
    ADD = 2

  def __init__(self, width: int, height: int, add_cost: float = -1., edit_cost: float =  -1.):
    super().__init__()

    self.add_cost = add_cost
    self.edit_cost = edit_cost

    # I'd really like to be able to have variable size images, not sure how to handle that using spaces
    self.width = width
    self.height = height
    self.rendered = np.zeros((self.height, self.width, 3))
    self.energy = 0.
    self.max_alpha = 0.1

    triangle_space = spaces.Box(low=0, high=1, shape=(10,))

    self.observation_space = spaces.Dict({"error": spaces.Box(low=-1, high=1, shape=(self.width, self.height)),
                                          "triangles": spaces.Sequence(triangle_space),
                                          "energy": spaces.Box(low=0, high=np.inf)})
    self.action_space = spaces.Dict(
      {"data": spaces.Dict({"index": spaces.Discrete(10), "triangle": triangle_space}), "op": spaces.Discrete(3)})

  def step(
    self, action: ActType
  ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
    """
    Possible actions:
      - stop. We're done, terminate the episode.
      - add. Add a new triangle
      - edit. Edit an existing triangle.

    What is the true evaluation here?



    Action space:
     - A dictionary
      - 'data': Dict.
        - 'edit': 'index': int, 'delta': delta on the selected triangle
        - 'add': 'triangle': the triangle array
      - 'op': The operation to perform, string, one of 'stop', 'add', 'edit'.

    :param action:
    :return:
    """
    terminated = False
    truncated = False
    reward_components = {}

    # if you don't have enough energy for the action you can't do it. No penalty for trying, but it's a wasted turn.
    # There should be a constant penalty for turns.
    match self.Op(action['op']):
      case self.Op.STOP:
        terminated = True
      case self.Op.EDIT:
        if self.energy >= self.edit_cost:
          idx = action['data']['index']
          if idx < len(self.triangles):
            triangle = action['data']['triangle']
            # for this first pass I'm going to simplify things by making edit a full replace action
            self.triangles[idx] = Triangle.from_array(triangle)

          # tried to edit an invalid triangle, still going to pay a cost.
          reward_components["edit"] = self.edit_cost
          self.energy -= self.edit_cost
      case self.Op.ADD:
        if self.energy >= self.add_cost:
          self.triangles.append(Triangle.from_array(action['data']['triangle']))
          reward_components["add"] = self.add_cost
          self.energy -= self.add_cost
      case _:
        raise Exception(f"Invalid op {action['op']}")

    rendered = render_triangles(self.triangles, height=self.height, width=self.width)

    # rendered and target_image are in pixel space
    reward_components["reconstruction"] = np.square(self.target_image - rendered).sum() / (
      rendered.shape[0] * rendered.shape[1])

    self.rendered = rendered

    self.energy = np.maximum(self.energy, 0.0)

    if self.energy <= 0.:
      terminated = True

    reward = -1  # a constant step penalty
    for v in reward_components.values():
      reward += v

    return self._get_obs(), reward, terminated, truncated, {"rewards": reward_components}

  def reset(
    self,
    *,
    seed: int | None = None,
    options: dict[str, Any] | None = None,
  ) -> tuple[ObsType, dict[str, Any]]:
    super().reset(seed=seed, options=options)

    assert options
    assert 'target' in options
    self.target_image = options['target']

    # for now we'll start with a fixed amount of energy
    self.energy = 10.

    self.width = self.target_image.shape[1]
    self.height = self.target_image.shape[0]

    self.triangles = [Triangle.from_array(t) for t in options["triangles"]] if "triangles" in options else []
    self.rendered = render_triangles(self.triangles, self.width, self.height)

    obs = self._get_obs()
    return obs, {}

  def render(self) -> RenderFrame:
    return render_triangles(self.triangles, width=self.width, height=self.height)

  def _get_obs(self) -> Dict[str, Any]:
    """
    Observation should be a delta between the target image and the current render.
    Should also include a list of all the current triangles.
    :return:
    """
    diff = self.target_image - self.rendered

    return {"error": diff, "triangles": self.triangles, "energy": self.energy}


def render_triangles(triangles: List[Triangle], width: int, height: int) -> np.ndarray:
  # TODO: for now just rendering in white
  image = Image.new("RGB", (width, height), (255, 255, 255))
  draw = ImageDraw.Draw(image, "RGBA")

  def scale_int(value: float, max_int: int) -> int:
    # while the call ends up being just as complicated, this way I make sure every call is consistent
    assert value <= 1.0
    return int(value * max_int)

  scale_rgba = partial(scale_int, max_int=255)

  for triangle in triangles:
    draw.polygon([(scale_int(v.x, width), scale_int(v.y, height)) for v in triangle.vertices],
                 (scale_rgba(triangle.r), scale_rgba(triangle.g), scale_rgba(triangle.b), scale_rgba(triangle.a)))

  del draw

  return np.array(image)


if __name__ == "__main__":

  width = 300
  height = 600
  channels = 3
  env = TriangleEnv(width=width, height=height)
  image = np.zeros(shape=(height, width, channels))

  init_obs, _ = env.reset(options={"target": image})
  for i in range(20):
    action = env.action_space.sample()
    obs, r, terminated, truncated, info = env.step(action)
    img = env.render()
    Image.fromarray(img).show()
