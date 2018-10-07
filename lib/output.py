import os
import numpy as np
import time
import requests
import os

dataset_url = "https://github.com/falcowinkler/n-maps-dataset/raw/master/n_maps.tfrecord"


def download_file(url, output_file):
    r = requests.get(url, stream=True)
    with open(output_file, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)


def compile_protobuf():
    download_file("https://raw.githubusercontent.com/falcowinkler/copicat/master/resources/proto/tile_data.proto",
                  "proto/tile_data.proto")

    os.system("protoc --python_out compiled_protobuf -I proto proto/*.proto")


def to_protobuf(generator_output, num_samples=8):
    # compile_protobuf()
    import compiled_protobuf.tile_data_pb2 as tile_data_pb2
    for sample_number, generated_level in enumerate(generator_output[:num_samples]):
        gen = np.argmax(generated_level, axis=2)
        tile_data = tile_data_pb2.TileData()
        data = np.round(gen).astype(np.int8)
        data_reshaped = np.reshape(data, [23 * 31])
        bytes_array = ''.join(chr(item) for item in data_reshaped.tolist())
        tile_data.raw_data = bytes_array
        filename = str(time.time()) + "_gen_sample_number_" + str(sample_number)
        f = open("out/proto/" + filename, "wb")
        f.write(tile_data.SerializeToString())
        f.close()
