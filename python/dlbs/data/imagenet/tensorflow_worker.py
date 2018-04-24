import os
import tensorflow as tf

class ImageCoder(object):
    """Helper class that provides TensorFlow image coding utilities."""
    
    # https://groups.google.com/forum/embed/?place=forum/torch7#!topic/torch7/fOSTXHIESSU
    png_file = 'n02105855_2933.JPEG'
    # https://github.com/cytsai/ilsvrc-cmyk-image-list
    cmyk_files = ['n01739381_1309.JPEG', 'n02077923_14822.JPEG',
                  'n02447366_23489.JPEG', 'n02492035_15739.JPEG',
                  'n02747177_10752.JPEG', 'n03018349_4028.JPEG',
                  'n03062245_4620.JPEG', 'n03347037_9675.JPEG',
                  'n03467068_12171.JPEG', 'n03529860_11437.JPEG',
                  'n03544143_17228.JPEG', 'n03633091_5218.JPEG',
                  'n03710637_5125.JPEG', 'n03961711_5286.JPEG',
                  'n04033995_2932.JPEG', 'n04258138_17003.JPEG',
                  'n04264628_27969.JPEG', 'n04336792_7448.JPEG',
                  'n04371774_5854.JPEG', 'n04596742_4225.JPEG',
                  'n07583066_647.JPEG', 'n13037406_4650.JPEG']

    @staticmethod
    def is_png(filename):
      """Determine if a file contains a PNG format image.
      Args:
        filename: string, path of the image file.
      Returns:
        boolean indicating if the image is a PNG.
      """
      return ImageCoder.png_file in filename
  
    @staticmethod
    def is_cmyk(filename):
        """Determine if file contains a CMYK JPEG format image.    
        Args:
            filename: string, path of the image file.
        Returns:
            boolean indicating if the image is a JPEG encoded with CMYK color space.
        """
        return filename.split('/')[-1] in ImageCoder.cmyk_files

    def __init__(self, shard, target_size=255):
        #
        import tensorflow as tf
        def _central_crop(image):
            # height = img_shape[0], width = img_shape[1]
            img_shape = tf.shape(image)
            crop_size = tf.minimum(img_shape[0], img_shape[1])
            return tf.image.crop_to_bounding_box(
                image,
                offset_height=(img_shape[0] - crop_size)/2,
                offset_width=(img_shape[1] - crop_size)/2,
                target_height=crop_size,
                target_width=crop_size
            )
        # Create a single Session to run all image coding calls.
        gpu_options = tf.GPUOptions(allow_growth=True, visible_device_list=str(shard))
        self._sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        
        # Initializes function that converts PNG to JPEG data.
        self._png_data = tf.placeholder(dtype=tf.string)
        image = tf.image.decode_png(self._png_data, channels=3)
        self._png_to_jpeg = tf.image.encode_jpeg(image, format='rgb', quality=100)

        # Initializes function that converts CMYK JPEG data to RGB JPEG data.
        self._cmyk_data = tf.placeholder(dtype=tf.string)
        image = tf.image.decode_jpeg(self._cmyk_data, channels=0)
        self._cmyk_to_rgb = tf.image.encode_jpeg(image, format='rgb', quality=100)

        # Initializes function that decodes RGB JPEG data. The 'decode_jpeg' operator
        # returns a tensor of type uint8.
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)   # Decode a JPEG-encoded image to a uint8 tensor.
        self._decode_jpeg = _central_crop(self._decode_jpeg)                           # uint8
        self._decode_jpeg = tf.image.resize_images(                                    # a 3-D float Tensor
            self._decode_jpeg,
            (target_size,target_size),
            align_corners=True
        )
        self._decode_jpeg = tf.cast(self._decode_jpeg, tf.uint8)                       # cast back to uint8 ([height, width, channels])

    def png_to_jpeg(self, image_data):
        return self._sess.run(self._png_to_jpeg,feed_dict={self._png_data: image_data})

    def cmyk_to_rgb(self, image_data):
        return self._sess.run(self._cmyk_to_rgb, feed_dict={self._cmyk_data: image_data})

    def decode_jpeg(self, image_data):
        image = self._sess.run(self._decode_jpeg, feed_dict={self._decode_jpeg_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image
    
    def decode_file(self, filename):
        # Read the image file.
        with tf.gfile.FastGFile(filename, 'r') as f:
            image_data = f.read()
        # Clean the dirty data.
        if self.is_png(filename):
            # 1 image is a PNG.
            print('Converting PNG to JPEG for %s' % filename)
            image_data = self.png_to_jpeg(image_data)
        elif self.is_cmyk(filename):
            # 22 JPEG images are in CMYK colorspace.
            print('Converting CMYK to RGB for %s' % filename)
            image_data = self.cmyk_to_rgb(image_data)
        # Decode the RGB JPEG.
        return self.decode_jpeg(image_data)


class TFRecord(object):
    @staticmethod
    def int64_feature(value):
        """Wrapper for inserting int64 features into Example proto."""
        if not isinstance(value, list):
            value = [value]
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    @staticmethod
    def bytes_feature(value):
        """Wrapper for inserting bytes features into Example proto."""
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @staticmethod
    def convert_to_example(filename, image, label, synset, human, height, width):
        """Build an Example proto for an example.
    
        Args:
            filename: string, path to an image file, e.g., '/path/to/example.JPG'
            image_buffer: string, JPEG encoding of RGB image
            label: integer, identifier for the ground truth for the network
            synset: string, unique WordNet ID specifying the label, e.g., 'n02323233'
            human: string, human-readable label, e.g., 'red fox, Vulpes vulpes'
            height: integer, image height in pixels
            width: integer, image width in pixels
        Returns:
            Example proto
        """
        colorspace = 'RGB'
        channels = 3
        image_format = 'tensor'

        example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': TFRecord.int64_feature(height),
            'image/width': TFRecord.int64_feature(width),
            'image/colorspace': TFRecord.bytes_feature(colorspace),
            'image/channels': TFRecord.int64_feature(channels),
            'image/class/label': TFRecord.int64_feature(label),
            'image/format': TFRecord.bytes_feature(image_format),
            'image/encoded': TFRecord.bytes_feature(image.tobytes()),
            'image/filename': TFRecord.bytes_feature(os.path.basename(filename)),    
            'image/class/synset': TFRecord.bytes_feature(synset),
            'image/class/text': TFRecord.bytes_feature(human),
            }))
        return example
