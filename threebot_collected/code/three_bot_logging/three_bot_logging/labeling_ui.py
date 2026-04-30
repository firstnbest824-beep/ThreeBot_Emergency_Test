"""Terminal Labeling Interface for Three-Bot.
# Three-Bot을 위한 터미널 기반 라벨링 인터페이스.

Provides a simple console UI for operators to label episodes 
immediately after completion.
# 에피소드(동작)가 끝난 직후 운영자가 즉시 성공/실패 여부를 라벨링할 수 있도록 
# 간단한 콘솔 UI를 제공합니다.

Output matches π_0.6 requirement:
# 출력 데이터는 π_0.6 학습 데이터 규격을 따릅니다:
- is_success (bool): 성공 여부
- failure_type (enum string): 실패 시 원인 유형
- retry_count (int): 재시도 횟수 (현재는 0으로 고정)

Author: Architect_Sonnet @ Nova Robotics
"""

import sys
import threading
import queue
from typing import Optional

# Windows compatibility for select
# 윈도우 환경 호환성을 위한 처리입니다.
if sys.platform == 'win32':
    import msvcrt
    # 윈도우에서는 msvcrt 라이브러리로 키 입력을 받습니다.

def get_char():
    """Read single character from stdin (Cross-platform)."""
    # 키보드로부터 한 글자를 읽어오는 크로스 플랫폼 함수입니다.
    if sys.platform == 'win32':
        # 윈도우: msvcrt.getch()를 사용해 엔터 없이 입력받고 디코딩합니다.
        return msvcrt.getch().decode('utf-8').lower()
    else:
        # 리눅스/맥: 표준 입력(sys.stdin)에서 1바이트를 읽습니다.
        return sys.stdin.read(1).lower()

def query_episode_label(episode_id: str) -> Optional[dict]:
    """
    BLOCKING function - use query_episode_label_async() in production.
    # 블로킹(Blocking) 함수입니다. (사용자 입력이 있을 때까지 프로그램이 멈춥니다)
    # 실제 사용 시에는 비동기 함수인 query_episode_label_async()를 권장합니다.
    
    Interactive prompt for episode labeling.
    # 사용자와 상호작용하며 에피소드 라벨을 묻습니다.
    
    Returns:
        dict with keys: is_success, failure_type, retry_count
        None if discarded
    """
    
    print(f"\n[🛑] Episode {episode_id} Finished.")
    # 에피소드 종료 알림
    print("Evaluate Outcome: [S]uccess / [F]ailure / [D]iscard? > ", end='', flush=True)
    # 평가 질문: 성공(S) / 실패(F) / 데이터 버림(D)?
    
    while True:
        choice = get_char()
        # 키 입력을 받습니다.
        print(choice) # Echo (입력한 글자를 화면에 보여줌)
        
        if choice == 's':
            # 성공(Success) 선택 시:
            return {
                'is_success': True,
                'failure_type': '',  # 실패 원인 없음
                'retry_count': 0
            }
            
        elif choice == 'f':
            # 실패(Failure) 선택 시:
            # 구체적인 실패 원인을 묻습니다.
            print("\nFailure Reason: [1]Collision [2]Timeout [3]Slip [4]Human Stop > ", end='', flush=True)
            
            reasons = {
                '1': 'collision',   # 충돌
                '2': 'timeout',     # 시간 초과
                '3': 'slip',        # 미끄러짐 (파지 실패)
                '4': 'manual_stop'  # 사람이 강제 정지
            }
            
            while True:
                r_choice = get_char()
                if r_choice in reasons:
                    print(reasons[r_choice])
                    return {
                        'is_success': False,
                        'failure_type': reasons[r_choice],
                        'retry_count': 0
                    }
                    
        elif choice == 'd':
            # 버림(Discard) 선택 시:
            print("\n[🗑️] Episode Discarded.")
            return None # None을 반환하여 데이터를 저장하지 않도록 합니다.
            
        else:
            # 잘못된 입력 시 다시 묻습니다.
            print("\nInvalid input. [S]uccess / [F]ailure / [D]iscard? > ", end='', flush=True)

def query_episode_label_async(
    episode_id: str, 
    timeout: float = 30.0
) -> Optional[dict]:
    """
    Non-blocking episode labeling with timeout.
    # 타임아웃이 있는 비동기(Non-blocking) 라벨링 함수입니다.
    # 메인 프로그램이 멈추지 않도록 별도 스레드에서 입력을 기다립니다.
    
    Args:
        episode_id: Episode identifier
        timeout: Max seconds to wait for user input (입력 대기 시간, 기본 30초)
    
    Returns:
        Metadata dict or None if timeout/discard
    """
    result_queue = queue.Queue()
    # 스레드 간 데이터 전달을 위한 큐(Queue) 생성
    
    def _blocking_query():
        """Runs in separate thread."""
        # 별도 스레드에서 실행될 내부 함수입니다.
        result = query_episode_label(episode_id) # 여기서 입력을 기다림 (블로킹)
        result_queue.put(result) # 결과를 큐에 넣음
    
    # Start thread
    # 스레드를 시작합니다 (Daemon=True로 설정하여 프로그램 종료 시 같이 종료되게 함).
    thread = threading.Thread(target=_blocking_query, daemon=True)
    thread.start()
    
    # Wait with timeout
    try:
        # 지정된 시간(timeout) 동안 큐에서 결과가 오기를 기다립니다.
        return result_queue.get(timeout=timeout)
    except queue.Empty:
        # 시간 초과 시 (Timeout):
        # 나중에 검토할 수 있도록 '리뷰 필요(needs_review)' 상태로 자동 저장합니다.
        print(f"\n[⏱️] Timeout: Episode {episode_id} marked for review")
        return {
            'is_success': False,
            'failure_type': 'timeout_no_label', # 라벨 없음
            'retry_count': 0,
            'needs_review': True # 추후 검토 필요 플래그
        }

if __name__ == "__main__":
    # Test async version
    # 이 파일을 직접 실행했을 때의 테스트 코드입니다.
    print("Testing async labeling with 5s timeout...")
    result = query_episode_label_async("TEST_001", timeout=5.0)
    print(f"Result: {result}")
