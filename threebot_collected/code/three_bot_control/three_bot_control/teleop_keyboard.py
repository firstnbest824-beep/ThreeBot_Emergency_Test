"""Keyboard Teleoperation for Three-Bot.
# Three-Bot을 위한 키보드 원격 제어 노드입니다.

Controls dual 3-DoF robots via keyboard input.
# 키보드 입력을 통해 두 개의 3-DoF(실제로는 5-DoF) 로봇을 제어합니다.
Output: /three_bot/teleop/action (Float32MultiArray)
# 출력: /three_bot/teleop/action 토픽으로 Float32MultiArray 메시지를 발행합니다.

"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Header
import sys
import threading

# Cross-platform key input
# 운영체제별(크로스 플랫폼) 키 입력 처리 로직입니다.
if sys.platform == 'win32':
    # 윈도우(Windows)일 경우:
    import msvcrt
    # 윈도우 전용 콘솔 입력 라이브러리인 msvcrt를 사용합니다.
else:
    # 리눅스(Linux)나 맥(Mac)일 경우:
    import termios
    import tty
    import select
    # 터미널 설정을 제어하기 위한 라이브러리들을 사용합니다.

msg = """
Three-Bot Teleoperation (5-DoF)
-------------------------------
Unit 1 (Left):      Unit 2 (Right):
   w                   i
 a s d               j k l

Global Z (Winch):
 q   e

Space: STOP ALL
CTRL-C: Quit
"""
# 사용자에게 보여줄 사용법(조작키) 메시지를 정의합니다.

class TeleopKeyboard(Node):
    # 키보드 제어 노드 클래스 정의 (Node 상속)
    def __init__(self):
        super().__init__('teleop_keyboard')
        # 부모 클래스(Node) 생성자를 호출하여 노드 이름을 'teleop_keyboard'로 설정합니다.
        
        self.publisher = self.create_publisher(
            Float32MultiArray, 
            '/three_bot/teleop/action', 
            10
        )
        # '/three_bot/teleop/action' 토픽으로 명령을 보낼 퍼블리셔를 생성합니다. (큐 사이즈: 10)
        self.heartbeat_pub = self.create_publisher(Header, 'teleop_heartbeat', 10)
        # 'teleop_heartbeat' 토픽으로 하트비트(생존 신호)를 보낼 퍼블리셔를 생성합니다.
        self.create_timer(0.1, self.publish_heartbeat)
        # 0.1초(10Hz)마다 publish_heartbeat 함수를 실행하는 타이머를 만듭니다.

        if sys.platform != 'win32':
            # 윈도우가 아닌 경우(리눅스 등), 현재 터미널 설정을 저장해둡니다. (나중에 복구하기 위해)
            self.settings = termios.tcgetattr(sys.stdin)
            
        self.speed = 0.1  # meters per step
        # 한 번 키를 눌렀을 때의 이동 속도(또는 거리)를 설정합니다.
        self.running = True
        # 루프 실행 여부를 제어하는 플래그입니다.
        
        # Current action state
        # 현재 액션 상태를 저장할 리스트를 초기화합니다. (5-DoF: [X1, Y1, X2, Y2, Z])
        self.actions = [0.0] * 5
        
        # Start input thread
        # 키 입력을 받는 함수(_input_loop)를 별도의 스레드로 실행합니다.
        # 이렇게 해야 메인 스레드가 멈추지 않고 계속 ROS 통신을 할 수 있습니다.
        self.thread = threading.Thread(target=self._input_loop)
        self.thread.start()
        
        print(msg)
        # 사용법 메시지를 화면에 출력합니다.

    def get_key(self):
        # 키보드 입력을 한 글자 읽어오는 함수입니다.
        if sys.platform == 'win32':
            # 윈도우의 경우:
            if msvcrt.kbhit():
                # 키보드가 눌렸는지 확인합니다.
                key = msvcrt.getch()
                # 눌린 키를 읽어옵니다.
                # 바이트(bytes)를 문자열(string)로 디코딩해서 반환합니다.
                return key.decode('utf-8')
            return ''
            # 눌린 키가 없으면 빈 문자열을 반환합니다.
        else:
            # 리눅스/맥의 경우:
            tty.setraw(sys.stdin.fileno())
            # 터미널을 Raw 모드로 변경합니다 (엔터 없이 바로 키 입력 받기 위해).
            rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
            # 입력이 들어왔는지 0.1초 동안 기다리며 확인합니다.
            if rlist:
                # 입력이 있으면 한 글자를 읽습니다.
                key = sys.stdin.read(1)
            else:
                key = ''
                # 없으면 빈 문자열.
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
            # 터미널 설정을 원래대로 되돌립니다.
            return key

    def publish_heartbeat(self):
        # 하트비트(생존 신호)를 전송하는 함수입니다.
        msg = Header()
        # Header 메시지 객체 생성.
        msg.stamp = self.get_clock().now().to_msg()
        # 현재 시간을 타임스탬프로 찍습니다.
        msg.frame_id = "teleop"
        # 프레임 ID 설정.
        self.heartbeat_pub.publish(msg)
        # 메시지를 보냅니다. (이걸로 다른 노드들이 "아, 키보드 노드가 아직 살아있구나" 하고 압니다.)

    def _input_loop(self):
        # 실제 키 입력을 계속 받아서 처리하는 메인 루프 함수입니다 (스레드에서 실행됨).
        try:
            while self.running and rclpy.ok():
                # running 플래그가 True이고 ROS가 정상 동작 중인 동안 계속 반복합니다.
                key = self.get_key()
                # 키 입력을 받습니다.
                
                # Reset actions
                # 매 반복마다 액션 값을 0으로 초기화합니다. (누르고 있을 때만 움직이게 하기 위해)
                self.actions = [0.0] * 5
                
                # 키에 따라 액션 값을 설정합니다.
                if key == 'w': self.actions[0] = self.speed  # W: X1 전진
                elif key == 's': self.actions[0] = -self.speed # S: X1 후진
                elif key == 'a': self.actions[1] = self.speed  # A: Y1 좌측
                elif key == 'd': self.actions[1] = -self.speed # D: Y1 우측
                
                elif key == 'i': self.actions[2] = self.speed  # I: X2 전진
                elif key == 'k': self.actions[2] = -self.speed # K: X2 후진
                elif key == 'j': self.actions[3] = self.speed  # J: Y2 좌측
                elif key == 'l': self.actions[3] = -self.speed # L: Y2 우측
                
                elif key == 'q': self.actions[4] = self.speed  # Q: Z 상승 (Winch Up)
                elif key == 'e': self.actions[4] = -self.speed # E: Z 하강 (Winch Down)
                
                # U/O keys removed (Z is now shared on Q/E - 5-DoF)
                
                elif key == ' ': # Emergency stop
                    # 스페이스바: 비상 정지 (모든 값 0)
                    self.actions = [0.0] * 5
                elif key == '\x03': # CTRL-C
                    # Ctrl+C: 종료
                    self.running = False
                    break
                
                # Publish if any key pressed
                # 키보드 입력이 있었을 때만 메시지를 발행합니다.
                if key != '':
                    msg = Float32MultiArray()
                    # 메시지 객체 생성.
                    msg.data = self.actions
                    # 데이터 채우기 (5개의 값).
                    self.publisher.publish(msg)
                    # ROS로 전송.
                    
        except Exception as e:
            # 에러 발생 시 출력합니다.
            print(e)
        finally:
            # 루프가 끝나면(종료 시) 마지막으로 정지 신호(0.0)를 보냅니다.
            msg = Float32MultiArray()
            msg.data = [0.0] * 5
            self.publisher.publish(msg)

    def stop(self):
        # 노드를 안전하게 종료하기 위한 함수입니다.
        self.running = False
        # 루프 종료 플래그 설정.
        self.thread.join()
        # 입력 스레드가 끝날 때까지 기다립니다.
        if sys.platform != 'win32':
            # 리눅스/맥의 경우 터미널 설정을 최종 복구합니다.
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)

def main(args=None):
    # 메인 함수입니다.
    rclpy.init(args=args)
    # ROS 2 초기화.
    node = TeleopKeyboard()
    # 노드 객체 생성.
    try:
        rclpy.spin(node)
        # 노드를 실행하고 메시지를 대기합니다.
    except KeyboardInterrupt:
        # Ctrl+C로 강제 종료 시 예외 처리.
        pass
    finally:
        # 종료 시 정리 작업.
        node.stop()
        # 스레드 종료 및 터미널 복구.
        node.destroy_node()
        # 노드 파괴.
        rclpy.shutdown()
        # ROS 2 종료.

if __name__ == '__main__':
    main()
