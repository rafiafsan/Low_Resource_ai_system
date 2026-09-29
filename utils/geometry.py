def get_side(point, line_start, line_end, reference_point):
    def cross_product(p):
        x1, y1 = line_start
        x2, y2 = line_end
        x, y = p
        return ((x2 - x1) * (y - y1) - (y2 - y1) * (x - x1))

    point_side = cross_product(point)
    reference_side = cross_product(reference_point)

    # Same side as inside_reference
    if point_side * reference_side >= 0:
        return "INSIDE"
    else:
        return "OUTSIDE"