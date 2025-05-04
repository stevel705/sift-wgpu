// tests/integration_test.rs
use image::{DynamicImage, GenericImageView, GrayImage, Luma}; // Добавили GenericImageViewuse sift::keypoints::KeyPoint; // Импортируем KeyPoint для доступа к полям
use sift::keypoints::KeyPoint;
use sift::sift::{load_image_dyn, Sift, DEFAULT_NUM_INTERVALS};
use std::path::Path;

// Ожидаемая длина SIFT дескриптора
const EXPECTED_DESCRIPTOR_LEN: usize = 128;
// Путь к основному тестовому изображению
const TEST_IMAGE_PATH: &str = "data/1.jpg";

/// Вспомогательная функция для проверки свойств ключевых точек
fn check_keypoint_properties(keypoints: &[KeyPoint], img_width: u32, img_height: u32) {
    let img_width_f = img_width as f32;
    let img_height_f = img_height as f32;

    for (i, kp) in keypoints.iter().enumerate() {
        // Координаты должны быть в пределах изображения (с небольшой погрешностью для интерполяции)
        assert!(
            kp.x >= -1.0 && kp.x <= img_width_f + 1.0,
            "KP[{}]: x coordinate out of bounds: {}",
            i,
            kp.x
        );
        assert!(
            kp.y >= -1.0 && kp.y <= img_height_f + 1.0,
            "KP[{}]: y coordinate out of bounds: {}",
            i,
            kp.y
        );
        // Размер (sigma) должен быть положительным
        assert!(
            kp.size > 0.0,
            "KP[{}]: size (sigma) is not positive: {}",
            i,
            kp.size
        );
        // Угол должен быть в пределах [-PI, PI]
        assert!(
            kp.angle >= -std::f32::consts::PI && kp.angle <= std::f32::consts::PI,
            "KP[{}]: angle out of bounds: {}",
            i,
            kp.angle
        );
        // Октава и слой должны быть в разумных пределах (зависит от параметров SIFT)
        // Например, октава >= 0
        assert!(
            kp.octave >= 0,
            "KP[{}]: octave is negative: {}",
            i,
            kp.octave
        );
        // Слой (после интерполяции) может быть немного за пределами [0, num_intervals+2], но не сильно
        // Точные границы зависят от деталей интерполяции, проверим на разумность
        assert!(
            kp.layer >= -2 && kp.layer <= (DEFAULT_NUM_INTERVALS as i32 + 4),
            "KP[{}]: layer is out of reasonable range: {}",
            i,
            kp.layer
        );
    }
}

#[test]
fn test_sift_on_real_image() {
    let image_path = Path::new(TEST_IMAGE_PATH);
    assert!(
        image_path.exists(),
        "Test image file not found at '{}'",
        TEST_IMAGE_PATH
    );

    let img_dyn = load_image_dyn(TEST_IMAGE_PATH).expect("Failed to load test image.");
    let (width, height) = img_dyn.dimensions();
    let sift = Sift::default();
    let (keypoints, descriptors) = sift.detect_and_compute(&img_dyn);

    // 1. Проверки на непустоту и совпадение количества
    assert!(
        !keypoints.is_empty(),
        "No keypoints found on {}",
        TEST_IMAGE_PATH
    );
    assert!(
        !descriptors.is_empty(),
        "No descriptors computed for {}",
        TEST_IMAGE_PATH
    );
    assert_eq!(keypoints.len(), descriptors.len(), "KP/Desc count mismatch");
    println!("Found {} keypoints on {}", keypoints.len(), TEST_IMAGE_PATH);

    // 2. Проверка длины дескрипторов
    for (i, desc) in descriptors.iter().enumerate() {
        assert_eq!(
            desc.len(),
            EXPECTED_DESCRIPTOR_LEN,
            "Desc[{}] length incorrect",
            i
        );
    }

    // 3. Проверка свойств точек
    check_keypoint_properties(&keypoints, width, height);
}

#[test]
fn test_sift_determinism() {
    let image_path = Path::new(TEST_IMAGE_PATH);
    assert!(
        image_path.exists(),
        "Test image file not found at '{}'",
        TEST_IMAGE_PATH
    );
    let img_dyn = load_image_dyn(TEST_IMAGE_PATH).expect("Failed to load test image.");
    let sift = Sift::default();

    // Первый запуск
    let (keypoints1, descriptors1) = sift.detect_and_compute(&img_dyn);
    // Второй запуск
    let (keypoints2, descriptors2) = sift.detect_and_compute(&img_dyn);

    // Сравниваем результаты
    // Используем PartialEq, который мы добавили для KeyPoint
    assert_eq!(
        keypoints1.len(),
        keypoints2.len(),
        "Keypoint counts differ between runs."
    );
    assert_eq!(
        descriptors1.len(),
        descriptors2.len(),
        "Descriptor counts differ between runs."
    );

    // Поэлементное сравнение (может быть чувствительно к float, но для детерминизма должно работать)
    // Сортировка необязательна, если порядок гарантирован, но добавим на всякий случай,
    // если вдруг порядок изменится в будущем (хотя не должен).
    // Для сортировки KeyPoint нужен Ord, которого нет. Сравним так.
    for i in 0..keypoints1.len() {
        assert_eq!(
            keypoints1[i], keypoints2[i],
            "Keypoint at index {} differs between runs.",
            i
        );
        // Сравнение float векторов
        assert_eq!(
            descriptors1[i].len(),
            descriptors2[i].len(),
            "Descriptor[{}] lengths differ",
            i
        );
        for j in 0..descriptors1[i].len() {
            assert!(
                (descriptors1[i][j] - descriptors2[i][j]).abs() < 1e-6,
                "Descriptor[{}][{}] differs: {} vs {}",
                i,
                j,
                descriptors1[i][j],
                descriptors2[i][j]
            );
        }
    }
    println!("SIFT results are deterministic on {}", TEST_IMAGE_PATH);
}

#[test]
fn test_sift_on_small_image() {
    // Создаем маленькое изображение (например, 20x15)
    let width = 20;
    let height = 15;
    // Создадим простой градиент, чтобы были хоть какие-то особенности
    let small_gray_img = GrayImage::from_fn(width, height, |x, _| Luma([(x * 10) as u8]));
    let img_dyn = DynamicImage::ImageLuma8(small_gray_img);

    let sift = Sift::default();
    let (keypoints, descriptors) = sift.detect_and_compute(&img_dyn);

    // На таком маленьком изображении может не найтись точек, или найтись очень мало
    // Мы не будем утверждать, что они *должны* найтись, но проверим базовые свойства, если нашлись.
    if !keypoints.is_empty() {
        println!("Found {} keypoints on small image.", keypoints.len());
        assert_eq!(
            keypoints.len(),
            descriptors.len(),
            "KP/Desc count mismatch on small image"
        );
        for (i, desc) in descriptors.iter().enumerate() {
            assert_eq!(
                desc.len(),
                EXPECTED_DESCRIPTOR_LEN,
                "Desc[{}] length incorrect on small image",
                i
            );
        }
        check_keypoint_properties(&keypoints, width, height);
    } else {
        println!("No keypoints found on small image (which might be expected).");
        assert!(
            descriptors.is_empty(),
            "Found descriptors without keypoints on small image"
        );
    }
}

#[test]
fn test_sift_on_no_feature_image() {
    // Создаем изображение одного цвета (например, 100x100 серого)
    let width = 100;
    let height = 100;
    let uniform_gray: GrayImage = GrayImage::from_pixel(width, height, Luma([128u8]));
    let img_dyn = DynamicImage::ImageLuma8(uniform_gray);

    let sift = Sift::default();
    let (keypoints, descriptors) = sift.detect_and_compute(&img_dyn);

    // На абсолютно однородном изображении не должно быть найдено ключевых точек SIFT
    assert!(
        keypoints.is_empty(),
        "Keypoints found on a uniform color image."
    );
    assert!(
        descriptors.is_empty(),
        "Descriptors found on a uniform color image."
    );
    println!("Correctly found no keypoints on uniform image.");
}

#[test]
fn test_sift_high_contrast_threshold() {
    // ... загрузка изображения ...
    let img_dyn = load_image_dyn(TEST_IMAGE_PATH).expect("Failed to load test image.");

    // Получаем дефолтные значения
    let sift_default = Sift::default();
    let (keypoints_default, _) = sift_default.detect_and_compute(&img_dyn);
    let default_count = keypoints_default.len();
    assert!(
        default_count > 0,
        "Default SIFT found no keypoints, cannot test high threshold effect."
    );

    // Создаем SIFT с высоким порогом, используя дефолтные значения для остальных параметров
    let high_contrast_threshold = 1.0; // Очень высокий
    let sift_high_thresh = Sift::new(
        sift_default.sigma,          // Берем из default
        sift_default.num_octaves,    // Берем из default
        sift_default.num_intervals,  // Берем из default
        sift_default.assumed_blur,   // Берем из default
        high_contrast_threshold,     // Используем наше значение
        sift_default.edge_threshold, // Берем из default
    );
    let (keypoints_high, descriptors_high) = sift_high_thresh.detect_and_compute(&img_dyn);
    // ... остальные проверки ...
    let high_thresh_count = keypoints_high.len();
    println!(
        "Default threshold found {} keypoints, High threshold found {}.",
        default_count, high_thresh_count
    );
    assert!(
        high_thresh_count < default_count / 2 || high_thresh_count == 0,
        "High contrast threshold did not significantly reduce keypoint count ({} vs default {}).",
        high_thresh_count,
        default_count
    );
    assert_eq!(
        keypoints_high.len(),
        descriptors_high.len(),
        "KP/Desc count mismatch (high thresh)"
    );
    if !keypoints_high.is_empty() {
        let (width, height) = img_dyn.dimensions();
        check_keypoint_properties(&keypoints_high, width, height);
        for desc in descriptors_high {
            assert_eq!(
                desc.len(),
                EXPECTED_DESCRIPTOR_LEN,
                "Desc length incorrect (high thresh)"
            );
        }
    }
}
