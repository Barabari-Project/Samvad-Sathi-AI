from allosaurus.app import read_recognizer

model = read_recognizer()
detected_str = model.recognize("pizza.wav", timestamp=False)
# detected = detected_str.split()  # Split into list


print(detected_str.replace(' ',''))
