import cv2


def calculate_new_size(new_wide, height, wide):
    new_wide = int(new_wide)
    f = float(new_wide) / wide
    new_size = (new_wide, int(height * f))

    return new_size


def resize_image(image, wide=500, scale=1.0, is_scaled=True):
    if is_scaled:
        new_wide = int(image.shape[1] * scale)
    else:
        new_wide = wide

    new_size = calculate_new_size(new_wide, image.shape[0], image.shape[1])
    resized_image = cv2.resize(image,
                               new_size,
                               interpolation=cv2.INTER_AREA)
    return resized_image


def rotate_image(ratio, image):
    (h, w) = image.shape[:2]
    center = (h / 2, w / 2)
    prepared_object = cv2.getRotationMatrix2D(center, ratio, 1.0)
    rotated_image = cv2.warpAffine(image, prepared_object, (w, h))
    return rotated_image


def flip_image(image, mode=0):
    return cv2.flip(image, mode)


def find_color(image, low_color):
    high_color = (255, 255, 255)
    only_object = cv2.inRange(image, low_color, high_color)
    return only_object


def run(path_to_image):
    image = cv2.imread(path_to_image)

    resized_image = resize_image(image=image, is_scaled=False)

    cv2.imshow("Resized image", resized_image)

    cropped_image = resized_image[570:670, 140:250]

    cv2.imshow("Cropped image", cropped_image)

    resized_cropped_image = resize_image(image=cropped_image,
                                         scale=3.0)
    cv2.imshow("x3 scale cropped image", resized_cropped_image)

    cv2.imshow("Rotation on 45", rotate_image(image=resized_image, ratio=45))
    cv2.imshow("Rotation on 90", rotate_image(image=resized_image, ratio=90))
    cv2.imshow("Rotation on 120", rotate_image(image=resized_image, ratio=120))

    new_resized_cropped_image = resize_image(image=cropped_image, scale=0.5)
    cv2.imshow("Rotation minimized image on 120", rotate_image(image=new_resized_cropped_image, ratio=120))

    cv2.imshow("Flip horizontally", flip_image(image=cropped_image, mode=0))
    cv2.imshow("Flip vertically", flip_image(image=cropped_image, mode=1))
    cv2.imshow("Flip horizontally and vertically", flip_image(image=cropped_image, mode=-1))

    cv2.imshow("Yellow", find_color(resized_cropped_image, low_color=(0, 150, 150)))
    cv2.waitKey(0)


if __name__ == '__main__':
    run("../resources/urban-view-with-cars-on-the-street.jpg")
