import sys, time
from pynput import keyboard # 키 입력 감지용
from global_variables import frameConfig, tracked_bboxes, resultBbox, droneControl, modeConfig

# Tello controlling threads: 추적된 xywh 좌표를 바탕으로 드론 조종

def up_down(e):
    global modeConfig, tracked_bboxes, frameConfig, resultBbox, droneControl    # 쓰레드에서 사용할 글로벌 변수
    while True:
        if modeConfig[1]:  # 수동 조작 모드(modeConfig[1]: manualMode)
            def on_press(key):
                global droneControl
                if key == keyboard.Key.up:  # up키가 눌린 경우
                    droneControl[2] = 40    # droneControl[2] = up_downValue
                elif key == keyboard.Key.down:  # down키가 눌린 경우
                    droneControl[2] = -40   # droneControl[2] = up_downValue
                
            def on_release(key):
                global droneControl
                if key == keyboard.Key.up or key == keyboard.Key.down:  # up이나 down키를 뗀 경우
                    droneControl[2] = 0 # droneControl[2] = up_downValue

                if key == keyboard.Key.space or key == keyboard.Key.esc or key == keyboard.Key.enter:
                    # Stop listener
                    print("Stopped listener")
                    return False
                
            with keyboard.Listener(
                on_press=on_press,
                on_release=on_release) as listener:
                    listener.join()

        else:   # 자동 추적 모드
            if(frameConfig[1]/2 - 40 < resultBbox[1] and resultBbox[1] < frameConfig[1]/2 + 40):  # 추적 대상이 가로 기준 화면의 가운데에 있는 경우
                # frameConfig[1]: frameHeight / resultBbox[1]: cy (추적 결과 박스의 중심 y좌표)
                droneControl[2] = 0 # droneControl[2] = up_downValue
            else:   # 추적 대상이 위아래로 치우친 경우
                droneControl[2] = (frameConfig[1]/2 - resultBbox[1])/4   # 유동 velocity값 적용    # droneControl[2] = up_downValue

        if modeConfig[0]:    # 메인 쓰레드에서 modeConfig[0](=exitFlag)를 True로 만들면 쓰레드 종료
            print("[DEBUG] up_down_thread exit.")
            sys.exit(0)

        time.sleep(0.05)    # 메인 쓰레드에 CPU타임을 보장하기 위한 지연

def yaw(e):
    global modeConfig, tracked_bboxes, frameConfig, resultBbox, droneControl # 쓰레드에서 사용할 글로벌 변수
    while True:
        if modeConfig[1]:  # 수동 조작 모드(modeConfig[1]: manualMode)
            def on_press(key):
                global droneControl
                if key == keyboard.Key.right:   # 키보드에서 오른쪽 방향키를 누른 경우
                    droneControl[0] = 100   # droneControl[0] = yawValue
                elif key == keyboard.Key.left:  # 키보드에서 왼쪽 방향키를 누른 경우
                    droneControl[0] = -100  # droneControl[0] = yawValue
            def on_release(key):
                global droneControl
                if key == keyboard.Key.right or key == keyboard.Key.left:   # 키보드에서 좌우 방향키를 뗀 경우
                    droneControl[0] = 0     # droneControl[0] = yawValue

                if key == keyboard.Key.space or key == keyboard.Key.esc or key == keyboard.Key.enter:
                    # Stop listener
                    return False
                
            with keyboard.Listener(
                on_press=on_press,
                on_release=on_release) as listener:
                    listener.join()

        else:   # 자동 추적 모드
            if(frameConfig[0]/2 - 40 < resultBbox[0] and resultBbox[0] < frameConfig[0]/2 + 40):  # 추적 대상이 가로 기준 화면의 가운데에 있는 경우
                # frameConfig[0]: frameWidth / resultBbox[0]: cx (추적 결과 박스의 중심 x좌표)
                droneControl[0] = 0 # droneControl[0] = yawValue
            else:   # 추적 대상이 양 옆으로 치우친 경우
                droneControl[0] = -(frameConfig[0]/2 - resultBbox[0])/8   # 유동 velocity값 적용    # droneControl[0] = yawValue
        
        if modeConfig[0]:    # 메인 쓰레드에서 modeConfig[0](=exitFlag)를 True로 만들면 쓰레드 종료
            print("[DEBUG] yaw_thread exit.")
            sys.exit(0)

        time.sleep(0.05)    # 메인 쓰레드에 CPU타임을 보장하기 위한 지연

def for_back(e):
    global modeConfig, tracked_bboxes, frameConfig, resultBbox, droneControl # 쓰레드에서 사용할 글로벌 변수
    while True:
        if modeConfig[1]:  # 수동 조작 모드(modeConfig[1]: manualMode)
            def on_press(key):
                global droneControl
                try:
                    if key.char == 'w':
                        droneControl[1] = 40    # droneControl[1]: for_backValue
                    elif key.char == 's':
                        droneControl[1] = -40   # droneControl[1]: for_backValue
                except AttributeError:  # 키보드에서 알파벳 외의 다른 특수키를 누른 경우 AttributeError가 발생함
                    None
            def on_release(key):
                global droneControl
                try:
                    if key.char == 'w' or key.char == 's':
                        droneControl[1] = 0     # droneControl[1]: for_backValue
                except AttributeError:
                    None

                if key == keyboard.Key.space or key == keyboard.Key.esc or key == keyboard.Key.enter:
                    # Stop listener
                    return False
                
            with keyboard.Listener(
                on_press=on_press,
                on_release=on_release) as listener:
                    listener.join()
            
        else:   # 자동 추적 모드
            if (resultBbox[2] < int(frameConfig[0] / 5)):  # 박스가 너무 작아진 경우 # resultBbox[2]: w (추적 결과 박스의 폭) / frameConfig[0]: frameWidth
                droneControl[1] = 30

            elif (resultBbox[2] > int(frameConfig[0] / 3) - 25):   # 박스가 너무 커진 경우
                droneControl[1] = -30
            
            else:
                droneControl[1] = 0
        
        if modeConfig[0]:    # 메인 쓰레드에서 modeConfig[0](=exitFlag)를 True로 만들면 쓰레드 종료
            print("[DEBUG] for_back_thread exit.")
            sys.exit(0)

        time.sleep(0.05)    # 메인 쓰레드에 CPU타임을 보장하기 위한 지연

def left_right(e):
    global modeConfig
    while True:
        if modeConfig[1]:  # 수동 조작 모드(modeConfig[1]: manualMode)
            def on_press(key):
                global droneControl
                try:
                    if key.char == 'd':
                        droneControl[3] = 40    # droneControl[3]: left_rightValue
                    elif key.char == 'a':
                        droneControl[3] = -40   # droneControl[3]: left_rightValue
                except AttributeError:  # 키보드에서 알파벳 외의 다른 특수키를 누른 경우 AttributeError가 발생함
                    None
            def on_release(key):
                global droneControl
                try:
                    if key.char == 'd' or key.char == 'a':
                        droneControl[3] = 0     # droneControl[3]: left_rightValue
                except AttributeError:
                    None

                if key == keyboard.Key.space or key == keyboard.Key.esc or key == keyboard.Key.enter:
                    # Stop listener
                    return False
                
            with keyboard.Listener(
                on_press=on_press,
                on_release=on_release) as listener:
                    listener.join()

        # left_right는 자동 추적 모드에서 사용하지 않음
        if modeConfig[0]:    # 메인 쓰레드에서 modeConfig[0](=exitFlag)를 True로 만들면 쓰레드 종료
            print("[DEBUG] left_right_thread exit.")
            sys.exit(0)

        time.sleep(0.05)    # 메인 쓰레드에 CPU타임을 보장하기 위한 지연

def intensity_control(e):   # 조작 강도 조절 함수
    '''
    Press 'Shift' key to control the drone faster

    Press 'Ctrl' key to control the drone slower
    '''
    global modeConfig
    while True:
        if modeConfig[1]:  # 수동 조작 모드(modeConfig[1]: manualMode)
            def on_press(key):
                global droneControl
                if key == keyboard.Key.shift:
                    droneControl[4] = 2.0   # droneControl[4]: intensityValue
                elif key == keyboard.Key.ctrl:
                    droneControl[4] = 0.5   # droneControl[4]: intensityValue
            def on_release(key):
                global droneControl
                if key == keyboard.Key.shift or key == keyboard.Key.ctrl:
                    droneControl[4] = 1.0   # droneControl[4]: intensityValue

                if key == keyboard.Key.space or key == keyboard.Key.esc or key == keyboard.Key.enter:
                    # Stop listener
                    return False
                
            with keyboard.Listener(
                on_press=on_press,
                on_release=on_release) as listener:
                    listener.join()

        # left_right는 자동 추적 모드에서 사용하지 않음
        
        if modeConfig[0]:    # 메인 쓰레드에서 modeConfig[0](=exitFlag)를 True로 만들면 쓰레드 종료
            print("[DEBUG] intensity_control_thread exit.")
            sys.exit(0)

        time.sleep(0.05)    # 메인 쓰레드에 CPU타임을 보장하기 위한 지연
    