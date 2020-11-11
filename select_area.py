# MIT License

# Copyright (c) 2016 Akshay Chavan

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import cv2


class Rectangle:

    x = None
    y = None
    w = None
    h = None

    def print_it(self):
        print(str(self.x) + ',' + str(self.y) + ',' + str(self.w) + ',' + str(self.h))


class DragRectangle:

    # Limits on the canvas
    keepWithin = Rectangle()
    # To store rectangle
    outRect = Rectangle()
    # To store rectangle anchor point
    # Here the rect class object is used to store
    # the distance in the x and y direction from
    # the anchor point to the top-left and the bottom-right corner
    anchor = Rectangle()
    # Selection marker size
    sBlk = 4
    # Whether initialized or not
    initialized = False

    # Image
    image = None

    # Window Name
    window_name = ""

    # Return flag
    return_flag = False

    # FLAGS
    # Rect already present
    active = False
    # Drag for rect resize in progress
    drag = False
    # Marker flags by positions
    TL = False
    TM = False
    TR = False
    LM = False
    RM = False
    BL = False
    BM = False
    BR = False
    hold = False


def init(drag_object, img, window_name, window_width, window_height):

    # Image
    drag_object.image = img

    # Window name
    drag_object.window_name = window_name

    # Limit the selection box to the canvas
    drag_object.keepWithin.x = 0
    drag_object.keepWithin.y = 0
    drag_object.keepWithin.w = window_width
    drag_object.keepWithin.h = window_height

    # Set rect to zero width and height
    drag_object.outRect.x = 0
    drag_object.outRect.y = 0
    drag_object.outRect.w = 0
    drag_object.outRect.h = 0


def drag_rectangle(event, x, y, flags, drag_object):

    if x < drag_object.keepWithin.x:
        x = drag_object.keepWithin.x

    if y < drag_object.keepWithin.y:
        y = drag_object.keepWithin.y

    if x > (drag_object.keepWithin.x + drag_object.keepWithin.w - 1):
        x = drag_object.keepWithin.x + drag_object.keepWithin.w - 1

    if y > (drag_object.keepWithin.y + drag_object.keepWithin.h - 1):
        y = drag_object.keepWithin.y + drag_object.keepWithin.h - 1

    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_down(x, y, drag_object)

    if event == cv2.EVENT_LBUTTONUP:
        mouse_up(x, y, drag_object)

    if event == cv2.EVENT_MOUSEMOVE:
        mouse_move(x, y, drag_object)

    if event == cv2.EVENT_LBUTTONDBLCLK:
        mouse_double_click(x, y, drag_object)


def rectangle_point(pX, pY, rX, rY, rW, rH):

    if rX <= pX <= (rX + rW) and rY <= pY <= (rY + rH):
        return True
    else:
        return False


def mouse_double_click(eX, eY, drag_object):

    if drag_object.active:

        if rectangle_point(eX, eY, drag_object.outRect.x, drag_object.outRect.y, drag_object.outRect.w, drag_object.outRect.h):
            drag_object.return_flag = True
            cv2.destroyWindow(drag_object.window_name)


def mouse_down(eX, eY, drag_object):

    if drag_object.active:

        if rectangle_point(eX, eY, drag_object.outRect.x - drag_object.sBlk,
                           drag_object.outRect.y - drag_object.sBlk,
                           drag_object.sBlk * 2, drag_object.sBlk * 2):
            drag_object.TL = True
            return

        if rectangle_point(eX, eY, drag_object.outRect.x + drag_object.outRect.w - drag_object.sBlk,
                           drag_object.outRect.y - drag_object.sBlk,
                           drag_object.sBlk * 2, drag_object.sBlk * 2):
            drag_object.TR = True
            return

        if rectangle_point(eX, eY, drag_object.outRect.x - drag_object.sBlk,
                           drag_object.outRect.y + drag_object.outRect.h - drag_object.sBlk,
                           drag_object.sBlk * 2, drag_object.sBlk * 2):
            drag_object.BL = True
            return

        if rectangle_point(eX, eY, drag_object.outRect.x + drag_object.outRect.w - drag_object.sBlk,
                           drag_object.outRect.y + drag_object.outRect.h - drag_object.sBlk,
                           drag_object.sBlk * 2, drag_object.sBlk * 2):
            drag_object.BR = True
            return

        if rectangle_point(eX, eY, drag_object.outRect.x + drag_object.outRect.w / 2 - drag_object.sBlk,
                           drag_object.outRect.y - drag_object.sBlk,
                           drag_object.sBlk * 2, drag_object.sBlk * 2):
            drag_object.TM = True
            return

        if rectangle_point(eX, eY, drag_object.outRect.x + drag_object.outRect.w / 2 - drag_object.sBlk,
                           drag_object.outRect.y + drag_object.outRect.h - drag_object.sBlk,
                           drag_object.sBlk * 2, drag_object.sBlk * 2):
            drag_object.BM = True
            return

        if rectangle_point(eX, eY, drag_object.outRect.x - drag_object.sBlk,
                           drag_object.outRect.y + drag_object.outRect.h / 2 - drag_object.sBlk,
                           drag_object.sBlk * 2, drag_object.sBlk * 2):
            drag_object.LM = True
            return

        if rectangle_point(eX, eY, drag_object.outRect.x + drag_object.outRect.w - drag_object.sBlk,
                           drag_object.outRect.y + drag_object.outRect.h / 2 - drag_object.sBlk,
                           drag_object.sBlk * 2, drag_object.sBlk * 2):
            drag_object.RM = True
            return

        # This has to be below all of the other conditions
        if rectangle_point(eX, eY, drag_object.outRect.x, drag_object.outRect.y, drag_object.outRect.w, drag_object.outRect.h):
            drag_object.anchor.x = eX - drag_object.outRect.x
            drag_object.anchor.w = drag_object.outRect.w - drag_object.anchor.x
            drag_object.anchor.y = eY - drag_object.outRect.y
            drag_object.anchor.h = drag_object.outRect.h - drag_object.anchor.y
            drag_object.hold = True

            return

    else:
        drag_object.outRect.x = eX
        drag_object.outRect.y = eY
        drag_object.drag = True
        drag_object.active = True
        return


def mouse_move(eX, eY, drag_object):

    if drag_object.drag & drag_object.active:
        drag_object.outRect.w = eX - drag_object.outRect.x
        drag_object.outRect.h = eY - drag_object.outRect.y
        clear_canvas_and_draw(drag_object)
        return

    if drag_object.hold:
        drag_object.outRect.x = eX - drag_object.anchor.x
        drag_object.outRect.y = eY - drag_object.anchor.y

        if drag_object.outRect.x < drag_object.keepWithin.x:
            drag_object.outRect.x = drag_object.keepWithin.x

        if drag_object.outRect.y < drag_object.keepWithin.y:
            drag_object.outRect.y = drag_object.keepWithin.y

        if (drag_object.outRect.x + drag_object.outRect.w) > (drag_object.keepWithin.x + drag_object.keepWithin.w - 1):
            drag_object.outRect.x = drag_object.keepWithin.x + drag_object.keepWithin.w - 1 - drag_object.outRect.w

        if (drag_object.outRect.y + drag_object.outRect.h) > (drag_object.keepWithin.y + drag_object.keepWithin.h - 1):
            drag_object.outRect.y = drag_object.keepWithin.y + drag_object.keepWithin.h - 1 - drag_object.outRect.h

        clear_canvas_and_draw(drag_object)
        return

    if drag_object.TL:
        drag_object.outRect.w = (drag_object.outRect.x + drag_object.outRect.w) - eX
        drag_object.outRect.h = (drag_object.outRect.y + drag_object.outRect.h) - eY
        drag_object.outRect.x = eX
        drag_object.outRect.y = eY
        clear_canvas_and_draw(drag_object)
        return

    if drag_object.BR:
        drag_object.outRect.w = eX - drag_object.outRect.x
        drag_object.outRect.h = eY - drag_object.outRect.y
        clear_canvas_and_draw(drag_object)
        return

    if drag_object.TR:
        drag_object.outRect.h = (drag_object.outRect.y + drag_object.outRect.h) - eY
        drag_object.outRect.y = eY
        drag_object.outRect.w = eX - drag_object.outRect.x
        clear_canvas_and_draw(drag_object)
        return

    if drag_object.BL:
        drag_object.outRect.w = (drag_object.outRect.x + drag_object.outRect.w) - eX
        drag_object.outRect.x = eX
        drag_object.outRect.h = eY - drag_object.outRect.y
        clear_canvas_and_draw(drag_object)
        return

    if drag_object.TM:
        drag_object.outRect.h = (drag_object.outRect.y + drag_object.outRect.h) - eY
        drag_object.outRect.y = eY
        clear_canvas_and_draw(drag_object)
        return

    if drag_object.BM:
        drag_object.outRect.h = eY - drag_object.outRect.y
        clear_canvas_and_draw(drag_object)
        return

    if drag_object.LM:
        drag_object.outRect.w = (drag_object.outRect.x + drag_object.outRect.w) - eX
        drag_object.outRect.x = eX
        clear_canvas_and_draw(drag_object)
        return

    if drag_object.RM:
        drag_object.outRect.w = eX - drag_object.outRect.x
        clear_canvas_and_draw(drag_object)
        return


def mouse_up(eX, eY, drag_object):

    drag_object.drag = False
    disable_resize_buttons(drag_object)
    straighten_up_rectangle(drag_object)
    if drag_object.outRect.w == 0 or drag_object.outRect.h == 0:
        drag_object.active = False

    clear_canvas_and_draw(drag_object)


def disable_resize_buttons(drag_object):

    drag_object.TL = drag_object.TM = drag_object.TR = False
    drag_object.LM = drag_object.RM = False
    drag_object.BL = drag_object.BM = drag_object.BR = False
    drag_object.hold = False


def straighten_up_rectangle(drag_object):

    if drag_object.outRect.w < 0:
        drag_object.outRect.x = drag_object.outRect.x + drag_object.outRect.w
        drag_object.outRect.w = -drag_object.outRect.w

    if drag_object.outRect.h < 0:
        drag_object.outRect.y = drag_object.outRect.y + drag_object.outRect.h
        drag_object.outRect.h = -drag_object.outRect.h


def clear_canvas_and_draw(drag_object):

    # Draw
    tmp = drag_object.image.copy()
    cv2.rectangle(tmp, (drag_object.outRect.x, drag_object.outRect.y),
                  (drag_object.outRect.x + drag_object.outRect.w,
                   drag_object.outRect.y + drag_object.outRect.h), (0, 255, 0), 2)
    draw_select_markers(tmp, drag_object)
    cv2.imshow(drag_object.window_name, tmp)
    cv2.waitKey()


def draw_select_markers(image, drag_object):

    # Top-Left
    cv2.rectangle(image,
                  (int(drag_object.outRect.x - drag_object.sBlk),
                   int(drag_object.outRect.y - drag_object.sBlk)),
                  (int(drag_object.outRect.x - drag_object.sBlk + drag_object.sBlk * 2),
                   int(drag_object.outRect.y - drag_object.sBlk + drag_object.sBlk * 2)),
                  (0, 255, 0), 2)
    # Top-Right
    cv2.rectangle(image,
                  (int(drag_object.outRect.x + drag_object.outRect.w - drag_object.sBlk),
                   int(drag_object.outRect.y - drag_object.sBlk)),
                  (int(drag_object.outRect.x + drag_object.outRect.w - drag_object.sBlk + drag_object.sBlk * 2),
                   int(drag_object.outRect.y - drag_object.sBlk + drag_object.sBlk * 2)),
                  (0, 255, 0), 2)
    # Bottom-Left
    cv2.rectangle(image,
                  (int(drag_object.outRect.x - drag_object.sBlk),
                   int(drag_object.outRect.y + drag_object.outRect.h - drag_object.sBlk)),
                  (int(drag_object.outRect.x - drag_object.sBlk + drag_object.sBlk * 2),
                   int(drag_object.outRect.y + drag_object.outRect.h - drag_object.sBlk + drag_object.sBlk * 2)),
                  (0, 255, 0), 2)
    # Bottom-Right
    cv2.rectangle(image,
                  (int(drag_object.outRect.x + drag_object.outRect.w - drag_object.sBlk),
                   int(drag_object.outRect.y + drag_object.outRect.h - drag_object.sBlk)),
                  (int(drag_object.outRect.x + drag_object.outRect.w - drag_object.sBlk + drag_object.sBlk * 2),
                   int(drag_object.outRect.y + drag_object.outRect.h - drag_object.sBlk + drag_object.sBlk * 2)),
                  (0, 255, 0), 2)

    # Top-Mid
    cv2.rectangle(image,
                  (int(drag_object.outRect.x + drag_object.outRect.w / 2 - drag_object.sBlk),
                   int(drag_object.outRect.y - drag_object.sBlk)),
                  (int(drag_object.outRect.x + drag_object.outRect.w / 2 - drag_object.sBlk + drag_object.sBlk * 2),
                   int(drag_object.outRect.y - drag_object.sBlk + drag_object.sBlk * 2)),
                  (0, 255, 0), 2)
    # Bottom-Mid
    cv2.rectangle(image,
                  (int(drag_object.outRect.x + drag_object.outRect.w / 2 - drag_object.sBlk),
                   int(drag_object.outRect.y + drag_object.outRect.h - drag_object.sBlk)),
                  (int(drag_object.outRect.x + drag_object.outRect.w / 2 - drag_object.sBlk + drag_object.sBlk * 2),
                   int(drag_object.outRect.y + drag_object.outRect.h - drag_object.sBlk + drag_object.sBlk * 2)),
                  (0, 255, 0), 2)
    # Left-Mid
    cv2.rectangle(image,
                  (int(drag_object.outRect.x - drag_object.sBlk),
                   int(drag_object.outRect.y + drag_object.outRect.h / 2 - drag_object.sBlk)),
                  (int(drag_object.outRect.x - drag_object.sBlk + drag_object.sBlk * 2),
                   int(drag_object.outRect.y + drag_object.outRect.h / 2 - drag_object.sBlk + drag_object.sBlk * 2)),
                  (0, 255, 0), 2)
    # Right-Mid
    cv2.rectangle(image,
                  (int(drag_object.outRect.x + drag_object.outRect.w - drag_object.sBlk),
                   int(drag_object.outRect.y + drag_object.outRect.h / 2 - drag_object.sBlk)),
                  (int(drag_object.outRect.x + drag_object.outRect.w - drag_object.sBlk + drag_object.sBlk * 2),
                   int(drag_object.outRect.y + drag_object.outRect.h / 2 - drag_object.sBlk + drag_object.sBlk * 2)),
                  (0, 255, 0), 2)
