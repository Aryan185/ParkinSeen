import ffmpeg

# Input MP3 file and output WAV file
input_file = "aud_latest.wav"
output_file = "output_file.wav"

# Run ffmpeg command
ffmpeg.input(input_file).output(output_file).run()