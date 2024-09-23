import serial
import argparse
from pythonosc import udp_client

# parser.add_argument("--ip_head", default="192.168.77.220",
def ArgParser():
    parser = argparse.ArgumentParser(description="description of the program")
    parser.add_argument("--ip_head", default="192.168.8.139",
        help="The ip of the OSC server")
    parser.add_argument("--port_head", type=int, default=22220,
        help="The port the OSC server is listening on")
    parser.add_argument("--ip_avater", default="127.0.0.1",
        help="The ip of the OSC server")
    parser.add_argument("--port_avater", type=int, default=5005,
        help="The port the OSC server is listening on")
    return parser.parse_args() 

def main(args):
    vowel_text = "euoeuoeuoeuoeuo"
    input_text = "テストテスト"
    speak_time = 5.0
    client_head  = udp_client.SimpleUDPClient(args.ip_head, args.port_head)
    # client_avater = udp_client.SimpleUDPClient(args.ip_avater, args.port_avater)
    # メッセージ送信
    client_head.send_message("/firebarion/voice/command", [vowel_text, speak_time])
    # client_avater.send_message("/firebarion/voice/command ", input_text,speak_time) 

if __name__ == "__main__":
    args = ArgParser()
    main(args)
