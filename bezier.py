import jax
import jax.numpy as np

import plotly.graph_objects as go
import plotly.express as px


@jax.vmap
def line(a, b, num_t=512):
  return np.linspace(a, b, num=num_t)


def lerp(a, b):
  assert a.shape[1] == b.shape[1]
  c = np.linspace(0, 1, num=a.shape[1])[np.newaxis, :, np.newaxis]
  return a * (1 - c) + b * c


def bezier(*points):
  if len(points) == 1:
    raise RuntimeError('This function requires >=2 points')
  elif len(points) == 2:
    return line(points[0], points[1])
  else:
    return lerp(bezier(*points[:-1]), bezier(*points[1:]))


def raster(coords, widths=None, colors=None, dim=128, background=np.array([0, 0, 0])):
  # coords: [batch_size, num_strokes, num_t, 2] (coordinate)
  # widths: [batch_size, num_strokes, num_t, 1] (scalar)
  # colors: [batch_size, num_strokes, num_t, 3] (rgb)

  assert coords.ndim == 4
  if widths is None: # default w=1
    widths = np.ones([*coords.shape[:-1], 1], dtype=np.float32) * 1
  else:
    assert widths.ndim == 4
  widths = widths / dim

  if colors is None: # default color is black
    colors = np.zeros([*coords.shape[:-1], 3], dtype=np.float32)
  else:
    assert widths.ndim == 4

  batch_size = coords.shape[0]
  grid = np.stack(np.meshgrid(np.arange(0, dim), np.arange(0, dim)), axis=-1) / dim
  grid = np.tile(grid[np.newaxis], [batch_size, 1, 1, 1])

  flat_grid = np.reshape(grid, [batch_size, dim * dim, 2])
  dists = flat_grid[:, np.newaxis, np.newaxis] - coords[..., np.newaxis, :] # [batch_size, num_strokes, num_t, flat_grid, 2]
  dists = np.sqrt(np.sum(dists ** 2, axis=-1)) # [batch_size, num_strokes, num_t, flat_grid]
  closest_stroke = np.argmin(np.min(dists, axis=2), axis=1)
  closest_t = np.argmin(np.min(dists, axis=1), axis=1)
  
  # get per-coord style -- must collect along stroke and t dimensions (vmap is helpful here)
  widths = jax.vmap(lambda a, b: np.take(a, b, axis=0))(widths, closest_stroke)
  widths = jax.vmap(lambda a, b: np.take(a, b, axis=1), in_axes=[1, 1], out_axes=1)(widths, closest_t)
  widths = widths[..., 0, 0]

  colors = jax.vmap(lambda a, b: np.take(a, b, axis=0))(colors, closest_stroke)
  colors = jax.vmap(lambda a, b: np.take(a, b, axis=1), in_axes=[1, 1], out_axes=1)(colors, closest_t)
  colors = colors[..., 0, :]

  # get alpha channel
  alpha = np.min(dists, axis=[1, 2])
  alpha = np.min(dists, axis=[1, 2]) - widths
  alpha = 1 - (alpha * dim)

  # compositing
  alpha = np.where(alpha < 0, 0, alpha)
  alpha = np.where(alpha > 1, 1, alpha)
  paint = alpha[..., np.newaxis] * colors + (1 - np.tile(alpha[..., np.newaxis], [1, 1, 3])) * background
  paint = np.reshape(paint, [batch_size, dim, dim, 3])

  return paint


def plot_bezier_check():
  points = [
    np.array([[0, 0], [0.2, 0.2]]),
    np.array([[0, 1], [0.5, 0.6]]),
    np.array([[1, 0], [0.1, 0.6]]),
    np.array([[1, 1], [0.8, 0.9]]),
  ]
  
  curve = bezier(*points)
  res1 = line(points[0], points[1])
  res2 = line(points[1], points[2])
  res3 = lerp(res1, res2)

  res4 = line(points[1], points[2])
  res5 = line(points[2], points[3])
  res6 = lerp(res4, res5)

  res7 = lerp(res3, res6)

  fig = go.Figure()
  fig.add_trace(go.Scatter(x=res1[0, :, 0], y=res1[0, :, 1], name='linear1'))
  fig.add_trace(go.Scatter(x=res2[0, :, 0], y=res2[0, :, 1], name='linear2'))
  fig.add_trace(go.Scatter(x=res3[0, :, 0], y=res3[0, :, 1], name='quad1'))
  fig.add_trace(go.Scatter(x=res4[0, :, 0], y=res4[0, :, 1], name='linear3'))
  fig.add_trace(go.Scatter(x=res5[0, :, 0], y=res5[0, :, 1], name='linear4'))
  fig.add_trace(go.Scatter(x=res6[0, :, 0], y=res6[0, :, 1], name='quad2'))
  fig.add_trace(go.Scatter(x=res7[0, :, 0], y=res7[0, :, 1], name='cubic'))
  fig.add_trace(go.Scatter(x=curve[0, :, 0], y=curve[0, :, 1], name='curve'))
  # `cubic` and `curve` should be the same
  fig.show()


def raster_check():
  points = [
    np.array([[0, 0], [0.2, 0.2]]),
    np.array([[0, 1], [0.5, 0.6]]),
    np.array([[1, 0], [0.1, 0.6]]),
    np.array([[1, 1], [0.8, 0.9]]),
  ]
  widths = line(
    np.array([[0], [1]]),
    np.array([[3], [0]])
  )
  colors = line(
    np.array([[1, 1, 0], [0, 1, 1]]),
    np.array([[1, 0, 1], [0, 1, 0]])
  )

  curve = bezier(*points)
  img = raster(
    curve[np.newaxis],
    widths=widths[np.newaxis],
    colors=colors[np.newaxis])
  fig = px.imshow(img[0])
  fig.show()
