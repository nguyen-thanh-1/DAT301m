import os
import time
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import urllib.request

# Định nghĩa các layer theo yêu cầu
CONTENT_LAYERS = ['block5_conv2']
STYLE_LAYERS = [
    'block1_conv1',
    'block2_conv1',
    'block3_conv1',
    'block4_conv1',
    'block5_conv1'
]

# Các tham số tối ưu (theo hướng dẫn)
CONTENT_WEIGHT = 2.5e-8
STYLE_WEIGHT = 1.0e-6
TV_WEIGHT = 1.0e-6
ITERATIONS = 700

def load_img(path_to_img, target_size=(224, 224)):
    """Đọc ảnh từ file và thay đổi kích thước."""
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    # Scale ảnh sao cho cạnh lớn nhất bằng với max dimension của target_size, 
    # nhưng theo yêu cầu bài toán là resize thẳng về 224x224 (hoặc tương tự)
    img = tf.image.resize(img, target_size)
    # convert_image_dtype đưa ảnh về [0, 1]. vgg19.preprocess_input yêu cầu [0, 255]
    img = img * 255.0 
    img = img[tf.newaxis, :]
    return img

def preprocess_image(img):
    """Tiền xử lý ảnh cho VGG19 (RGB -> BGR, Zero-centering)."""
    return tf.keras.applications.vgg19.preprocess_input(img)

def deprocess_image(processed_img):
    """Phục hồi ảnh sau khi đã qua preprocess_input để hiển thị."""
    x = processed_img.copy()
    if len(x.shape) == 4:
        x = np.squeeze(x, 0)
    
    # Cộng lại ImageNet mean
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    
    # Đổi BGR về RGB
    x = x[:, :, ::-1]

    # Cắt các giá trị để nằm trong khoảng 0-255
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def vgg_layers(layer_names):
    """Khởi tạo mô hình VGG19 và trích xuất các layer cần thiết."""
    # Tải mô hình VGG19 (include_top=False vì ta không cần phần phân loại)
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    
    outputs = [vgg.get_layer(name).output for name in layer_names]
    model = tf.keras.Model([vgg.input], outputs)
    return model

def gram_matrix(input_tensor):
    """Tính ma trận Gram để đo lường độ tương quan giữa các kênh (Style)."""
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result / num_locations

class StyleContentModel(tf.keras.models.Model):
    """Mô hình đóng gói để trả về Content và Style features."""
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.vgg = vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs):
        # Yêu cầu input đã được preprocess_input (float32, [0,255])
        outputs = self.vgg(inputs)
        style_outputs, content_outputs = (outputs[:self.num_style_layers], 
                                          outputs[self.num_style_layers:])

        # Tính Gram Matrix cho các Style layers
        style_outputs = [gram_matrix(style_output) for style_output in style_outputs]

        # Tạo dictionary chứa output
        content_dict = {content_name:value 
                        for content_name, value in zip(self.content_layers, content_outputs)}
        style_dict = {style_name:value
                      for style_name, value in zip(self.style_layers, style_outputs)}
        
        return {'content': content_dict, 'style': style_dict}

def compute_loss(outputs, style_targets, content_targets):
    """Tính tổng hàm loss dựa trên Content, Style."""
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    
    # 1. Style Loss
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name] - style_targets[name])**2) 
                           for name in style_outputs.keys()])
    style_loss *= STYLE_WEIGHT / len(STYLE_LAYERS)

    # 2. Content Loss
    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name] - content_targets[name])**2) 
                             for name in content_outputs.keys()])
    content_loss *= CONTENT_WEIGHT / len(CONTENT_LAYERS)
    
    loss = style_loss + content_loss
    return loss

# --- PHẦN THỰC THI (TRAINING LOOP) ---
def run_style_transfer(content_path, style_path):
    print("1. Đang nạp và xử lý ảnh...")
    content_image = load_img(content_path)
    style_image = load_img(style_path)

    # Khởi tạo mô hình
    print("2. Đang nạp mô hình VGG19...")
    extractor = StyleContentModel(STYLE_LAYERS, CONTENT_LAYERS)

    # Lấy các giá trị mục tiêu (Targets)
    style_targets = extractor(preprocess_image(style_image))['style']
    content_targets = extractor(preprocess_image(content_image))['content']

    # Khởi tạo bức ảnh bắt đầu (từ ảnh content nguyên bản)
    image = tf.Variable(content_image)

    # Tối ưu hóa: Dùng Adam với exponential decay (hoặc learning rate cố định)
    opt = tf.keras.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

    @tf.function()
    def train_step(image):
        with tf.GradientTape() as tape:
            # Chú ý: Cần preprocess image hiện tại trước khi đưa vào VGG19
            outputs = extractor(preprocess_image(image))
            
            # Tính Content & Style Loss
            loss = compute_loss(outputs, style_targets, content_targets)
            
            # 3. Total Variation Loss (Làm mịn, giảm nhiễu hạt)
            loss += TV_WEIGHT * tf.image.total_variation(image)

        # Tính đạo hàm và cập nhật ảnh
        grad = tape.gradient(loss, image)
        opt.apply_gradients([(grad, image)])
        
        # Clip giá trị pixel về 0-255 sau khi cập nhật
        image.assign(tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=255.0))
        return loss

    print("3. Bắt đầu quá trình tối ưu hóa...")
    start_time = time.time()

    # Tạo thư mục output nếu chưa có
    os.makedirs('output', exist_ok=True)
    
    # Chạy vòng lặp
    for i in range(1, ITERATIONS + 1):
        loss = train_step(image)
        
        if i % 100 == 0:
            elapsed = time.time() - start_time
            print(f"Bước {i}/{ITERATIONS} - Loss: {loss[0]:.4f} - Thời gian: {elapsed:.1f}s")
            
            # Lưu ảnh kết quả theo yêu cầu (100, 300, 500, 700)
            if i in [100, 300, 500, 700]:
                out_img = deprocess_image(image.numpy())
                out_path = f"output/result_iter_{i}.jpg"
                Image.fromarray(out_img).save(out_path)
                print(f"--> Đã lưu ảnh: {out_path}")

    print("Hoàn tất Style Transfer!")

def download_sample_images():
    """Tải một số ảnh mẫu trên mạng nếu chưa có sẵn ảnh."""
    content_url = 'https://upload.wikimedia.org/wikipedia/commons/d/d7/Green_Sea_Turtle_grazing_seagrass.jpg'
    style_url = 'https://upload.wikimedia.org/wikipedia/commons/0/0a/The_Great_Wave_off_Kanagawa.jpg'
    
    content_path = 'turtle_content.jpg'
    style_path = 'wave_style.jpg'
    
    if not os.path.exists(content_path):
        print("Đang tải ảnh Content mẫu...")
        urllib.request.urlretrieve(content_url, content_path)
    if not os.path.exists(style_path):
        print("Đang tải ảnh Style mẫu...")
        urllib.request.urlretrieve(style_url, style_path)
        
    return content_path, style_path

if __name__ == '__main__':
    # Bạn có thể thay đổi đường dẫn tới ảnh của riêng mình ở đây
    # Ví dụ:
    my_content = 'path_to_my_content.jpg'
    my_style = 'path_to_my_style.jpg'
    
    # Ở đây tôi dùng ảnh mẫu tự tải về nếu không tìm thấy ảnh
    print("Chuẩn bị dữ liệu...")
    content_path, style_path = download_sample_images()
    
    run_style_transfer(content_path, style_path)
