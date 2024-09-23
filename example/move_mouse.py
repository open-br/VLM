import serial
import argparse

def ArgParser():
    parser = argparse.ArgumentParser(description="description of the program")
    return parser.parse_args() 

def main(args):
    ser = serial.Serial("COM3", 115200)
    ser.write(b"@")
    input_text = input("閉じるならエンター")
    ser.write(b"#")
    input_text = input("終わるならエンター")
    ser.close()

if __name__ == "__main__":
    args = ArgParser()
    main(args)
