def get_distance(x1, y1, x2, y2):
    """
    Pythagorean distance between two points
    """

    distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

    return distance
