"""Render paired episode recordings to a headless MP4."""

import imageio.v2 as imageio
import numpy as np

# PIL renderer similar to LiftSim's original blue-building scene.
from PIL import Image, ImageDraw, ImageFont

PANEL_W, PANEL_H = 672, 800
FRAME_W, FRAME_H = PANEL_W * 2, PANEL_H
BLUE = (160, 183, 215)
LINE = (77, 73, 111)
SHAFT = (126, 150, 185)
CAR = (127, 144, 172)
DOOR = (155, 171, 196)
TEXT = (255, 255, 255)
FLOOR_Y0 = 710  # y-position of the center of floor 1

def font(sz, bold=False):
    f = '/usr/share/fonts/truetype/dejavu/DejaVuSans' + ('-Bold.ttf' if bold else '.ttf')
    try:
        return ImageFont.truetype(f, sz)
    except OSError:
        return ImageFont.load_default(size=sz)

FONT_13, FONT_15 = font(13), font(15)
FONT_18, FONT_22 = font(18, True), font(22, True)

def draw_person(d, x, y, tint=(224, 99, 78), scale=1.0):
    """Little standing human silhouette to represent a waiting passenger."""
    r = lambda n: max(1, round(n*scale))
    x, y = int(x), int(y)
    d.ellipse((x-r(4), y-r(18), x+r(4), y-r(10)), fill=(105, 64, 57))
    d.rectangle((x-r(5), y-r(10), x+r(5), y+r(1)), fill=tint)
    d.line((x-r(3), y, x-r(4), y+r(9)), fill=(53, 61, 109), width=r(3))
    d.line((x+r(3), y, x+r(4), y+r(9)), fill=(53, 61, 109), width=r(3))
    d.line((x-r(6), y-r(8), x-r(8), y-r(1)), fill=(224, 183, 152), width=r(2))
    d.line((x+r(6), y-r(8), x+r(8), y-r(1)), fill=(224, 183, 152), width=r(2))

def y_for_floor(floor, floor_count):
    gap = min(57, 513 / max(1, floor_count - 1))
    return FLOOR_Y0 - (float(floor) - 1) * gap

def draw_car(d, cx, car, floor_count, car_count):
    y = y_for_floor(car['floor'], floor_count)
    w = min(46, max(10, 270 // car_count))
    h = min(49, max(16, 470 // floor_count))
    x0, y0 = int(cx-w/2), int(y-h/2)
    d.rectangle((x0-2, y0-2, x0+w+2, y0+h+2), fill=(72, 80, 112))
    d.rectangle((x0, y0+9, x0+w, y0+h), fill=(73, 80, 103))
    door_open = min(1., max(0., car['door']))
    half = max(1, int((w-5)*(1-door_open)/2))
    if half > 0:
        d.rectangle((x0+2, y0+10, x0+2+half, y0+h-2), fill=DOOR, outline=(100, 113, 147))
        d.rectangle((x0+w-2-half, y0+10, x0+w-2, y0+h-2), fill=DOOR, outline=(100, 113, 147))
    d.rectangle((x0, y0, x0+w, y0+9), fill=(33, 35, 48))
    direction = car['direction']
    arrow = (114, 222, 69) if direction > 0 else (255, 200, 91) if direction < 0 else (114, 120, 132)
    if direction >= 0:
        d.polygon([(cx-5, y0+7),(cx, y0+2),(cx+5, y0+7)], fill=arrow)
    else:
        d.polygon([(cx-5, y0+2),(cx, y0+7),(cx+5, y0+2)], fill=arrow)
    if car['load']:
        d.text((cx, y0+20), str(car['load']), font=FONT_13, fill=TEXT, anchor='mm')

def draw_panel(img, x_offset, frame, title):
    d = ImageDraw.Draw(img)
    left = x_offset
    d.rectangle((left, 0, left+PANEL_W-1, PANEL_H-1), fill=BLUE)
    d.rectangle((left, 0, left+PANEL_W-1, 145), fill=(133, 163, 203))
    d.text((left+PANEL_W//2, 18), title, font=FONT_22, fill=TEXT, anchor='mt')
    d.text((left+40, 58), 'Waiting up', font=FONT_18, fill=TEXT)
    d.text((left+PANEL_W-180, 58), 'Waiting down', font=FONT_18, fill=TEXT)
    cars = frame['cars']
    floor_count = len(frame['up'])
    car_x = np.linspace(242, 461, len(cars)) if len(cars) > 1 else [PANEL_W // 2]
    for i, cx in enumerate(car_x):
        d.text((left+cx, 108), f'E{i+1}: {frame["cars"][i]["load"]} people',
               font=FONT_13, fill=TEXT, anchor='mt')
    # floor stripes + floors + hallway walls
    for floor in range(1, len(frame['up'])+1):
        y = int(y_for_floor(floor, floor_count))
        d.line((left+8, y+29, left+PANEL_W-8, y+29), fill=LINE, width=4)
        d.text((left+9, y-15), f'{floor:02}', font=FONT_13, fill=(54, 68, 99))
        up_n = frame['up'][floor-1]
        down_n = frame['down'][floor-1]
        for k in range(min(3, up_n)):
            draw_person(d, left+80+k*22, y+8, tint=(215, 106, 82), scale=0.75)
        for k in range(min(3, down_n)):
            draw_person(d, left+540+k*22, y+8, tint=(99, 76, 173), scale=0.75)
        if up_n > 3:
            d.text((left+83, y+20), f'+{up_n-3}', font=FONT_13, fill=(54, 68, 99))
        if down_n > 3:
            d.text((left+557, y+20), f'+{down_n-3}', font=FONT_13, fill=(54, 68, 99))
    for cx, car in zip(car_x, cars):
        draw_car(d, left + int(cx), car, floor_count, len(cars))
    d.rectangle((left+0, PANEL_H-52, left+PANEL_W-1, PANEL_H-1), fill=(34, 47, 74))
    delivered = frame['stats'].get('DeliveredPersons(10Minutes)', 0)
    d.text((left+12, PANEL_H-33), f'Step {frame["step"]:3}    Delivered {delivered:3}',
           font=FONT_15, fill=TEXT)
    d.text((left+PANEL_W-12, PANEL_H-33), f'Reward {frame["reward"]:.3f}',
           font=FONT_15, fill=TEXT, anchor='ra')

def render_frame(a, b, titles):
    img = Image.new('RGB', (FRAME_W, FRAME_H))
    draw_panel(img, 0, a, titles[0])
    draw_panel(img, PANEL_W, b, titles[1])
    return img




def write_video(path, left, right, titles, *, fps=5):
    if not left or not right or len(left) != len(right):
        raise ValueError("Video requires two complete recordings of equal length")
    with imageio.get_writer(path, fps=fps, codec="libx264", quality=7,
                            macro_block_size=16) as writer:
        for a, b in zip(left, right):
            writer.append_data(np.asarray(render_frame(a, b, titles)))
