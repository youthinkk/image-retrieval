def load_index(label):
    if label == "alley":
        return class_alley()
    elif label == "antlers":
        return class_antler()
    elif label == "baby":
        return class_baby()
    elif label == "balloons":
        return class_balloon()
    elif label == "beach":
        return class_beach()
    elif label == "bear":
        return class_bear()
    elif label == "birds":
        return class_bird()
    elif label == "boats":
        return class_boat()
    elif label == "cars":
        return class_car()
    elif label == "cat":
        return class_cat()
    elif label == "computer":
        return class_computer()
    elif label == "coral":
        return class_coral()
    elif label == "dog":
        return class_dog()
    elif label == "fish":
        return class_fish()
    elif label == "flags":
        return class_flag()
    elif label == "flowers":
        return class_flower()
    elif label == "horses":
        return class_horse()
    elif label == "leaf":
        return class_leaf()
    elif label == "plane":
        return class_plane()
    elif label == "rainbow":
        return class_rainbow()
    elif label == "rocks":
        return class_rock()
    elif label == "sign":
        return class_sign()
    elif label == "snow":
        return class_snow()
    elif label == "tiger":
        return class_tiger()
    elif label == "tower":
        return class_tower()
    elif label == "train":
        return class_train()
    elif label == "tree":
        return class_tree()
    elif label == "whales":
        return class_whale()
    elif label == "window":
        return class_window()
    elif label == "zebra":
        return class_zebra()

    return class_empty()


def class_empty():
    """
    For alley, baby, leaf, rainbow, rock and tree
    """
    return []


def class_alley():
    return [702, 990, 690, 686, 685, 701]


def class_baby():
    return [966, 805, 298, 909, 765, 296, 297]


def class_leaf():
    return [328, 327, 332]


def class_rainbow():
    return [712, 992, 235, 368, 366, 367]


def class_rock():
    return [363, 359, 360, 367, 719]


def class_tree():
    return [306, 683, 366, 690, 682, 330]


def class_antler():
    return [9, 12, 52, 57, 65, 81, 162]


def class_balloon():
    return [233]


def class_beach():
    return [363, 364, 367]


def class_bear():
    return [7, 61, 103, 163, 169, 209, 786]


def class_bird():
    return [i for i in xrange(383, 442)]


def class_boat():
    index = [i for i in xrange(235, 249)]
    index += [689]
    return index


def class_car():
    index = [264, 265]
    index += [i for i in xrange(267, 276)]
    index += [282, 285, 286]
    return index


def class_cat():
    return [8, 10, 55, 95, 174, 199, 201]


def class_computer():
    return [228, 510, 550, 551, 552, 869]


def class_coral():
    return [365, 648, 649, 886]


def class_dog():
    index = [15]
    index += [i for i in xrange(17, 22)]
    index += [i for i in xrange(25, 30)]
    index += [i for i in xrange(31, 34)]
    index += [36]
    index += [i for i in xrange(41, 44)]
    index += [i for i in xrange(45, 48)]
    index += [i for i in xrange(49, 52)]
    index += [56, 59, 60, 63, 64, 66]
    index += [i for i in xrange(68, 73)]
    index += [77, 79, 83]
    index += [i for i in xrange(86, 92)]
    index += [93, 94]
    index += [i for i in xrange(97, 100)]
    index += [i for i in xrange(105, 108)]
    index += [109, 110]
    index += [i for i in xrange(112, 120)]
    index += [i for i in xrange(123, 129)]
    index += [i for i in xrange(130, 135)]
    index += [i for i in xrange(139, 142)]
    index += [i for i in xrange(143, 147)]
    index += [i for i in xrange(148, 153)]
    index += [i for i in xrange(154, 157)]
    index += [158, 160, 161, 168]
    index += [i for i in xrange(170, 174)]
    index += [176, 177, 179, 180, 184, 187, 189, 192]
    index += [i for i in xrange(196, 199)]
    index += [200, 202, 204, 207, 208, 210, 211, 253]
    return index


def class_fish():
    index = [i for i in xrange(447, 458)]
    index += [498, 647]
    return index


def class_flag():
    return [995]


def class_flower():
    return [357, 358]


def class_horse():
    return [39, 293]


def class_plane():
    index = [i for i in xrange(230, 233)]
    index += [246, 503]
    return index


def class_sign():
    return [932]


def class_snow():
    return [278, 288, 590]


def class_tiger():
    return [76]


def class_tower():
    index = [685]
    index += [i for i in xrange(690, 693)]
    index += [933]
    return index


def class_train():
    return [256, 257, 262, 263, 287, 887]


def class_whale():
    return [6, 22]


def class_window():
    return [904, 912]


def class_zebra():
    return [80]
