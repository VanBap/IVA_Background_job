import sys

import ffmpeg  # noqa
import logging
import os

logging.basicConfig()
logger = logging.getLogger('app')


class MediaFileError(Exception):
    pass


def extract_thumbnail(video_file_path, thumbnail_file_path='', capture_time=0, is_video=True,
                      print_cmd_output=False):
    try:
        if is_video:
            pipeline = ffmpeg.input(video_file_path, ss=capture_time, skip_frame='nokey')
        else:
            pipeline = ffmpeg.input(video_file_path)

        if thumbnail_file_path:
            # write to file
            pipeline = pipeline.output(thumbnail_file_path, vframes=1)
        else:
            # write to bytes
            pipeline = pipeline.output('pipe:', vframes=1, format='image2', vcodec='mjpeg')

        # run
        log_level = 'debug' if print_cmd_output else 'warning'
        out_bytes, stdout = (
            pipeline
            .global_args('-loglevel', log_level)
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
        if print_cmd_output:
            logger.info(stdout)

        if thumbnail_file_path:
            if os.path.isfile(thumbnail_file_path):
                return thumbnail_file_path
        elif out_bytes:
            return out_bytes

        raise MediaFileError(f'Output data is empty. Detail log: f{stdout}')

    except ffmpeg.Error as ex:
        raise MediaFileError(ex.stderr.decode()) from ex


# run
if __name__ == '__main__':

    extract_thumbnail('rtsp://admin:abcd1234@10.124.68.222:554/Streaming/Channels/101', '/home/vbd-vanhk-l1-ubuntu/work/test011025.jpg')