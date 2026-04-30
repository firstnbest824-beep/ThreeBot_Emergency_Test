/* USER CODE BEGIN Header */
/**
 ******************************************************************************
 * @file           : main.c
 * @brief          : NOVA Firmware (Architecture Migrated)
 *
 * [역할 분리(중요)]
 *  - main.c      : wiring only
 *      * UART RX ISR(HAL 콜백)  : drivers/drv_uart1.c 가 "단독 소유"
 *      * while loop            : app_cmd_process(), app_fsm_update()
 *
 *  - app_cmd     : 명령 파싱/라우팅(예: ACT:WINCH IN 5000, PING 등)
 *  - app_fsm     : 실행 스케줄링/가드/상태 관리 + DONE/FAIL 전송
 *
 *  - svc_winch   : 릴레이 제어(Active-Low)
 *  - svc_cylinder: 릴레이 제어(Active-Low)
 *  - svc_stepper : 축 호밍 FSM(논블로킹 tick 기반)
 *
 *  - drivers/    : 저수준(GPIO/ALM/TIM PWM/UART 등)
 ******************************************************************************
 */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include <math.h>
#include <stdbool.h>
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */
/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

// ADC 설정
#define ADC_CHANNELS 4
#define ADC_RESOLUTION 4095.0f
#define ADC_CENTER 2048.0f
#define ADC_DEADZONE 30.0f    // 축소: 대각선 인식 개선
#define ADC_FAILSAFE_MIN 100  // Open-circuit 하한 (2.5%)
#define ADC_FAILSAFE_MAX 3995 // Open-circuit 상한 (97.5%)

// 모터 제어
#define MOTOR_MAX_SPEED 999 // TIM3 Period에 맞춤 (Prescaler 조정으로 5배 속도)
#define SLEW_RATE_LIMIT 0.1f
#define MOTOR_ENABLE_THRESHOLD 0.005f // 축소: 미세 제어 향상

// 타이밍
#define CONTROL_PERIOD_MS 20 // 50ms→20ms (50Hz): 응답속도 2.5배 향상
#define DEBOUNCE_TIME_MS 10

// ALM 신호 디바운싱
#define STALL_DEBOUNCE_COUNT 3 // 연속 3회 감지 시 스톨 인정

// Protocol v1.3: Implicit Mode Switching & Sum Checksum
#define PROTOCOL_HEADER1 0xAA
#define PROTOCOL_HEADER2 0x55
#define UART_RX_BUF_SIZE 8
#define UART_TX_BUF_SIZE 14
#define UART_TIMEOUT_MS 500
#define STALL_OVERRIDE_MS 200 // 스톨 감지 시 명령 차단 시간

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */
/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
ADC_HandleTypeDef hadc1;
DMA_HandleTypeDef hdma_adc1;

TIM_HandleTypeDef htim3;

UART_HandleTypeDef huart1;

/* USER CODE BEGIN PV */

// ADC DMA 버퍼 (DMA 최적화를 위한 4-byte alignment)
volatile uint16_t adc_buffer[ADC_CHANNELS] __attribute__((aligned(4))) = {0};

// 조이스틱 정규화 값
float joy1_x = 0.0f, joy1_y = 0.0f;
float joy2_x = 0.0f, joy2_y = 0.0f;
float prev_joy1_x = 0.0f, prev_joy1_y = 0.0f;
float prev_joy2_x = 0.0f, prev_joy2_y = 0.0f;

// Z축 버튼 상태
uint32_t z_up_last_time = 0, z_down_last_time = 0;
bool z_up_pressed = false, z_down_pressed = false;

bool emergency_stop_active = false;

// ========== 디버그용 전역 변수 (Live Expressions 모니터링용) ==========
volatile uint16_t debug_pwm1_duty = 0;
volatile uint16_t debug_pwm2_duty = 0;
volatile uint16_t debug_pwm3_duty = 0;
volatile uint16_t debug_pwm4_duty = 0;
volatile float debug_motor1_speed = 0.0f;
volatile float debug_motor2_speed = 0.0f;
volatile float debug_motor3_speed = 0.0f;
volatile float debug_motor4_speed = 0.0f;
volatile bool debug_motor1_enabled = false;
volatile bool debug_motor2_enabled = false;
volatile bool debug_motor3_enabled = false;
volatile bool debug_motor4_enabled = false;

// ========== 가상 엔코더 (Virtual Encoder) ==========
volatile float virtual_steps_accumulator[4] = {0.0f};

// ========== Hybrid Control State ==========
volatile bool is_auto_mode = false;
volatile uint32_t last_uart_rx_time = 0;
float auto_cmd_x1 = 0.0f, auto_cmd_y1 = 0.0f;
float auto_cmd_x2 = 0.0f, auto_cmd_y2 = 0.0f;
bool auto_cmd_winch_in = false;
bool auto_cmd_winch_out = false;

// UART RX Buffer
uint8_t uart_rx_buffer[UART_RX_BUF_SIZE];
/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_DMA_Init(void);
static void MX_USART1_UART_Init(void);
static void MX_TIM3_Init(void);
static void MX_ADC1_Init(void);
/* USER CODE BEGIN PFP */
void Joystick_Init(void);
void Joystick_Read(void);
float Joystick_Normalize(uint16_t raw_value);
float Joystick_ApplySlewRate(float current, float previous);
bool Button_Read_Debounced(GPIO_TypeDef *GPIOx, uint16_t GPIO_Pin,
                           uint32_t *last_time, bool *state);
void Motor_Control(void);
void Emergency_Stop_Check(void);
void UART_Send_Packet(void);
void Parse_UART_Packet(void);
void Motor_Enable(GPIO_TypeDef *GPIOx, uint16_t GPIO_Pin, bool enable);
static void uart1_tx(const uint8_t *data, uint16_t len);
/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

/* ============================================================================
 * 조이스틱 제어 함수 구현
 * ============================================================================
 */

/* 조이스틱 초기화 */
void Joystick_Init(void) {
  // ADC DMA 시작
  HAL_ADC_Start_DMA(&hadc1, (uint32_t *)adc_buffer, ADC_CHANNELS);

  // 모터 Enable 핀 비활성화 (초기 안전 상태)
  Motor_Enable(GPIOE, ENA_1_Pin, false);
  Motor_Enable(GPIOE, ENA_2_Pin, false);
  Motor_Enable(GPIOE, ENA_3_Pin, false);
  Motor_Enable(GPIOE, ENA_4_Pin, false);

  // PWM을 0으로 초기화
  __HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_1, 0);
  __HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_2, 0);
  __HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_3, 0);
  __HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_4, 0);
}

/* ADC 값을 -1.0 ~ +1.0으로 정규화 */
float Joystick_Normalize(uint16_t raw_value) {
  float centered = (float)raw_value - ADC_CENTER;

  // Deadzone 적용
  if (fabsf(centered) < ADC_DEADZONE) {
    return 0.0f;
  }

  // -1.0 ~ +1.0 범위로 정규화
  float normalized = centered / (ADC_RESOLUTION / 2.0f);

  // Clamp
  if (normalized > 1.0f)
    normalized = 1.0f;
  if (normalized < -1.0f)
    normalized = -1.0f;

  return normalized;
}

/* Slew Rate Limiter (급격한 변화 방지) */
float Joystick_ApplySlewRate(float current, float previous) {
  float delta = current - previous;

  if (delta > SLEW_RATE_LIMIT) {
    return previous + SLEW_RATE_LIMIT;
  } else if (delta < -SLEW_RATE_LIMIT) {
    return previous - SLEW_RATE_LIMIT;
  }

  return current;
}

/* 조이스틱 값 읽기 (Fail-Safe 포함) */
void Joystick_Read(void) {
  // Fail-Safe: 조이스틱 단선 감지
  for (int i = 0; i < ADC_CHANNELS; i++) {
    uint16_t raw = adc_buffer[i];
    if (raw < ADC_FAILSAFE_MIN || raw > ADC_FAILSAFE_MAX) {
      // 조이스틱 단선 의심 → 비상 정지
      emergency_stop_active = true;
      return;
    }
  }

  // ADC 값 정규화 (물리적 조이스틱 위치와 모터 매핑 일치)
  // 물리적 왼쪽 조이스틱 → joy1 → 왼쪽 모터(모터1, 모터4)
  float joy1_x_raw = Joystick_Normalize(adc_buffer[3]); // 물리적 왼쪽 VRx (PC3)
  float joy1_y_raw = Joystick_Normalize(adc_buffer[2]); // 물리적 왼쪽 VRy (PC2)

  // 물리적 오른쪽 조이스틱 → joy2 → 오른쪽 모터(모터2, 모터3)
  float joy2_x_raw =
      Joystick_Normalize(adc_buffer[0]); // 물리적 오른쪽 VRx (PC0)
  float joy2_y_raw =
      Joystick_Normalize(adc_buffer[1]); // 물리적 오른쪽 VRy (PC1)

  // 정규화된 값을 직접 사용 (Slew Rate 제거 - 즉시 반응!)
  joy1_x = joy1_x_raw;
  joy1_y = joy1_y_raw;
  joy2_x = joy2_x_raw;
  joy2_y = joy2_y_raw;

  // 이전 값 저장 (호환성 유지)
  prev_joy1_x = joy1_x;
  prev_joy1_y = joy1_y;
  prev_joy2_x = joy2_x;
  prev_joy2_y = joy2_y;
}

/* 디바운스가 적용된 버튼 읽기 */
bool Button_Read_Debounced(GPIO_TypeDef *GPIOx, uint16_t GPIO_Pin,
                           uint32_t *last_time, bool *state) {
  uint32_t now = HAL_GetTick();
  GPIO_PinState pin_state = HAL_GPIO_ReadPin(GPIOx, GPIO_Pin);

  // Active Low 버튼
  bool pressed = (pin_state == GPIO_PIN_RESET);

  if (pressed != *state) {
    if ((now - *last_time) >= DEBOUNCE_TIME_MS) {
      *state = pressed;
      *last_time = now;
    }
  }

  return *state;
}

/* 모터 속도 설정 */
void Motor_SetSpeed(uint32_t tim_channel, float speed) {
  // speed: -1.0 ~ +1.0
  float abs_speed = fabsf(speed);
  uint32_t pwm_value = (uint32_t)(abs_speed * MOTOR_MAX_SPEED);

  if (pwm_value > MOTOR_MAX_SPEED)
    pwm_value = MOTOR_MAX_SPEED;

  __HAL_TIM_SET_COMPARE(&htim3, tim_channel, pwm_value);
}

/* 모터 방향 설정 */
void Motor_SetDirection(GPIO_TypeDef *GPIOx, uint16_t GPIO_Pin, bool forward) {
  HAL_GPIO_WritePin(GPIOx, GPIO_Pin, forward ? GPIO_PIN_SET : GPIO_PIN_RESET);
}

/* 모터 활성화 (Active Low 드라이버용) */
void Motor_Enable(GPIO_TypeDef *GPIOx, uint16_t GPIO_Pin, bool enable) {
  // HB808C는 Active Low: enable=true일 때 0V(RESET) 출력하여 모터 ON
  HAL_GPIO_WritePin(GPIOx, GPIO_Pin, enable ? GPIO_PIN_RESET : GPIO_PIN_SET);
}

/* 비상 정지 체크 */
void Emergency_Stop_Check(void) {
  // BTN_STOP (PE1) 읽기 - Active Low
  GPIO_PinState e_stop = HAL_GPIO_ReadPin(GPIOE, BTN_STOP_Pin);

  if (e_stop == GPIO_PIN_RESET) {
    // E-Stop 활성화
    emergency_stop_active = true;

    // 모든 모터 즉시 정지
    Motor_Enable(GPIOE, ENA_1_Pin, false);
    Motor_Enable(GPIOE, ENA_2_Pin, false);
    Motor_Enable(GPIOE, ENA_3_Pin, false);
    Motor_Enable(GPIOE, ENA_4_Pin, false);

    // PWM 0
    __HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_1, 0);
    __HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_2, 0);
    __HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_3, 0);
    __HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_4, 0);

    // 윈치 정지 (Active Low - SET = OFF)
    HAL_GPIO_WritePin(GPIOD, WINCH_OUT_Pin, GPIO_PIN_SET);
    HAL_GPIO_WritePin(GPIOD, WINCH_IN_Pin, GPIO_PIN_SET);
  } else {
    emergency_stop_active = false;
  }
}

/* 모터 제어 */
void Motor_Control(void) {
  static bool homing_active = false;
  static bool motor_stalled_homing[4] = {false, false, false, false};
  static bool prev_sw_homing = false;

  // ========== ALM 신호 읽기 (PA0~PA3, Active Low) ==========
  bool alm_motor1 = (HAL_GPIO_ReadPin(GPIOA, GPIO_PIN_0) == GPIO_PIN_RESET);
  bool alm_motor2 = (HAL_GPIO_ReadPin(GPIOA, GPIO_PIN_1) == GPIO_PIN_RESET);
  bool alm_motor3 = (HAL_GPIO_ReadPin(GPIOA, GPIO_PIN_2) == GPIO_PIN_RESET);
  bool alm_motor4 = (HAL_GPIO_ReadPin(GPIOA, GPIO_PIN_3) == GPIO_PIN_RESET);

  // [Global Stall Protection]
  static uint8_t alm_debounce[4] = {0, 0, 0, 0};
  bool stall_active[4] = {false, false, false, false};
  bool raw_alms[4] = {alm_motor1, alm_motor2, alm_motor3, alm_motor4};

  for (int i = 0; i < 4; i++) {
    if (raw_alms[i]) {
      if (alm_debounce[i] < STALL_DEBOUNCE_COUNT)
        alm_debounce[i]++;
    } else {
      if (alm_debounce[i] > 0)
        alm_debounce[i]--;
    }
    if (alm_debounce[i] >= STALL_DEBOUNCE_COUNT)
      stall_active[i] = true;
  }

  // ========== 제어 버튼 읽기 (Active Low) ==========
  // CubeMX에서 설정한 Label(매크로) 사용
  bool sw_homing =
      (HAL_GPIO_ReadPin(HOMING_GPIO_Port, HOMING_Pin) == GPIO_PIN_RESET);
  // [Note] sw_z_up, sw_z_down은 아래에서 디바운싱 로직으로 대체됨
  bool sw_estop =
      (HAL_GPIO_ReadPin(BTN_STOP_GPIO_Port, BTN_STOP_Pin) == GPIO_PIN_RESET);

  // 1. 비상 정지
  if (sw_estop) {
    emergency_stop_active = true;
    homing_active = false;
  }

  if (emergency_stop_active)
    return;

  // ========== 1. Z축(윈치) 제어 (최우선 순위 및 디바운싱 적용) ==========
  // 수동 모드일 때만 물리 버튼에서 상태 업데이트 (Auto 모드면 통신값이 우선)
  if (!is_auto_mode) {
    Button_Read_Debounced(Z_UP_GPIO_Port, Z_UP_Pin, &z_up_last_time,
                          &z_up_pressed);
    Button_Read_Debounced(Z_DOWN_GPIO_Port, Z_DOWN_Pin, &z_down_last_time,
                          &z_down_pressed);
  }

  if (z_up_pressed && !z_down_pressed) {
    // 윈치 OUT (Z-UP) - Active Low
    HAL_GPIO_WritePin(GPIOD, WINCH_OUT_Pin, GPIO_PIN_RESET);
    HAL_GPIO_WritePin(GPIOD, WINCH_IN_Pin, GPIO_PIN_SET);
  } else if (z_down_pressed && !z_up_pressed) {
    // 윈치 IN (Z-DOWN) - Active Low
    HAL_GPIO_WritePin(GPIOD, WINCH_OUT_Pin, GPIO_PIN_SET);
    HAL_GPIO_WritePin(GPIOD, WINCH_IN_Pin, GPIO_PIN_RESET);
  } else {
    // 윈치 정지
    HAL_GPIO_WritePin(GPIOD, WINCH_OUT_Pin, GPIO_PIN_SET);
    HAL_GPIO_WritePin(GPIOD, WINCH_IN_Pin, GPIO_PIN_SET);
  }

  // 2. 호밍 트리거 (Rising Edge)
  if (sw_homing && !prev_sw_homing && !homing_active) {
    homing_active = true;
    motor_stalled_homing[0] = false;
    motor_stalled_homing[1] = false;
    motor_stalled_homing[2] = false;
    motor_stalled_homing[3] = false;
  }
  prev_sw_homing = sw_homing;

  // 3. 호밍 시퀀스 (스톨 감지)
  if (homing_active) {
    float homing_speed = 0.3f;

    // ALM 신호로 스톨 감지
    if (alm_motor1 && !motor_stalled_homing[0])
      motor_stalled_homing[0] = true;
    if (alm_motor2 && !motor_stalled_homing[1])
      motor_stalled_homing[1] = true;
    if (alm_motor3 && !motor_stalled_homing[2])
      motor_stalled_homing[2] = true;
    if (alm_motor4 && !motor_stalled_homing[3])
      motor_stalled_homing[3] = true;

    bool all_stalled = motor_stalled_homing[0] && motor_stalled_homing[1] &&
                       motor_stalled_homing[2] && motor_stalled_homing[3];

    if (!all_stalled) {
      // 아직 안된 모터만 이동
      if (!motor_stalled_homing[0]) {
        Motor_Enable(GPIOE, ENA_1_Pin, true);
        Motor_SetDirection(GPIOE, DIR_1_Pin, 0);
        Motor_SetSpeed(TIM_CHANNEL_1, homing_speed);
      } else {
        Motor_SetSpeed(TIM_CHANNEL_1, 0);
      }

      if (!motor_stalled_homing[1]) {
        Motor_Enable(GPIOE, ENA_2_Pin, true);
        Motor_SetDirection(GPIOE, DIR_2_Pin, 0);
        Motor_SetSpeed(TIM_CHANNEL_2, homing_speed);
      } else {
        Motor_SetSpeed(TIM_CHANNEL_2, 0);
      }

      if (!motor_stalled_homing[2]) {
        Motor_Enable(GPIOE, ENA_3_Pin, true);
        Motor_SetDirection(GPIOE, DIR_3_Pin, 0);
        Motor_SetSpeed(TIM_CHANNEL_3, homing_speed);
      } else {
        Motor_SetSpeed(TIM_CHANNEL_3, 0);
      }

      if (!motor_stalled_homing[3]) {
        Motor_Enable(GPIOE, ENA_4_Pin, true);
        Motor_SetDirection(GPIOE, DIR_4_Pin, 0);
        Motor_SetSpeed(TIM_CHANNEL_4, homing_speed);
      } else {
        Motor_SetSpeed(TIM_CHANNEL_4, 0);
      }
    } else {
      // 모두 완료 → OCP 리셋 (Non-blocking)
      static uint32_t reset_start_time = 0;

      if (reset_start_time == 0) {
        // 1) 리셋 시작: 모터 정지 및 Enable Off
        Motor_SetSpeed(TIM_CHANNEL_1, 0);
        Motor_SetSpeed(TIM_CHANNEL_2, 0);
        Motor_SetSpeed(TIM_CHANNEL_3, 0);
        Motor_SetSpeed(TIM_CHANNEL_4, 0);

        Motor_Enable(GPIOE, ENA_1_Pin, false);
        Motor_Enable(GPIOE, ENA_2_Pin, false);
        Motor_Enable(GPIOE, ENA_3_Pin, false);
        Motor_Enable(GPIOE, ENA_4_Pin, false);

        reset_start_time = HAL_GetTick();
        if (reset_start_time == 0)
          reset_start_time = 1; // 0 회피
        return;                 // Non-blocking: 다음 루프 대기
      }

      // 2) 50ms 대기 확인
      if (HAL_GetTick() - reset_start_time >= 50) {
        // 50ms 경과: 좌표 리셋 및 호밍 종료
        joy1_x = 0.0f;
        joy1_y = 0.0f;
        joy2_x = 0.0f;
        joy2_y = 0.0f;
        prev_joy1_x = 0;
        prev_joy1_y = 0;
        prev_joy2_x = 0;
        prev_joy2_y = 0;

        homing_active = false;
        reset_start_time = 0; // 초기화
      }
    }

    return;
  }

  // 4. 조이스틱 제어 (하드웨어 배선에 맞춰 수정됨)
  // Loop: 20ms (0.02s) -> Max steps/loop tracking
  const float MAX_STEPS_PER_LOOP = 100.0f;

  // X1 (joy1_x) → 모터1: PA6(CH1), PE8(DIR), PE12(ENA) ✅
  if (stall_active[0])
    joy1_x = 0.0f; // Safety Override
  if (fabsf(joy1_x) > MOTOR_ENABLE_THRESHOLD) {
    virtual_steps_accumulator[0] += joy1_x * MAX_STEPS_PER_LOOP;
  }
  Motor_Enable(GPIOE, ENA_1_Pin, fabsf(joy1_x) > MOTOR_ENABLE_THRESHOLD);
  Motor_SetDirection(GPIOE, DIR_1_Pin, joy1_x > 0);
  Motor_SetSpeed(TIM_CHANNEL_1, joy1_x);

  // Y1 (joy1_y) → 모터4 핀 사용: PB1(CH4), PE11(DIR), PE15(ENA) 🔧 왼쪽 Y!
  if (stall_active[3])
    joy1_y = 0.0f; // Safety Override (Motor 4)
  if (fabsf(joy1_y) > MOTOR_ENABLE_THRESHOLD) {
    virtual_steps_accumulator[3] += joy1_y * MAX_STEPS_PER_LOOP;
  }
  Motor_Enable(GPIOE, ENA_4_Pin, fabsf(joy1_y) > MOTOR_ENABLE_THRESHOLD);
  Motor_SetDirection(GPIOE, DIR_4_Pin,
                     joy1_y < 0); // 좌측 Y축 다시 반전 (원래대로)
  Motor_SetSpeed(TIM_CHANNEL_4, joy1_y);

  // X2 (joy2_x) → 모터2 핀 사용: PA7(CH2), PE9(DIR), PE13(ENA) - 왼쪽 모듈 X축
  if (stall_active[1])
    joy2_x = 0.0f; // Safety Override (Motor 2)
  if (fabsf(joy2_x) > MOTOR_ENABLE_THRESHOLD) {
    virtual_steps_accumulator[1] += joy2_x * MAX_STEPS_PER_LOOP;
  }
  Motor_Enable(GPIOE, ENA_2_Pin, fabsf(joy2_x) > MOTOR_ENABLE_THRESHOLD);
  Motor_SetDirection(GPIOE, DIR_2_Pin, joy2_x > 0); // 우측 X축 다시 반전
  Motor_SetSpeed(TIM_CHANNEL_2, joy2_x);

  // Y2 (joy2_y) → 모터3 핀 사용: PB0(CH3), PE10(DIR), PE14(ENA) 🔧 오른쪽 Y!
  if (stall_active[2])
    joy2_y = 0.0f; // Safety Override (Motor 3)
  if (fabsf(joy2_y) > MOTOR_ENABLE_THRESHOLD) {
    virtual_steps_accumulator[2] += joy2_y * MAX_STEPS_PER_LOOP;
  }
  Motor_Enable(GPIOE, ENA_3_Pin, fabsf(joy2_y) > MOTOR_ENABLE_THRESHOLD);
  Motor_SetDirection(GPIOE, DIR_3_Pin,
                     joy2_y > 0); // 우측 Y축 다시 반전
  Motor_SetSpeed(TIM_CHANNEL_3, joy2_y);

  // ========== 디버그 변수 업데이트 (물리적 채널 순서로 정렬) ==========
  debug_motor1_speed = joy1_x; // CH1 (PA6)
  debug_motor2_speed = joy2_x; // CH2 (PA7)
  debug_motor3_speed = joy2_y; // CH3 (PB0) - 오른쪽 Y축
  debug_motor4_speed = joy1_y; // CH4 (PB1) - 왼쪽 Y축

  debug_motor1_enabled = (fabsf(joy1_x) > MOTOR_ENABLE_THRESHOLD);
  debug_motor2_enabled = (fabsf(joy2_x) > MOTOR_ENABLE_THRESHOLD);
  debug_motor3_enabled = (fabsf(joy2_y) > MOTOR_ENABLE_THRESHOLD); // 수정
  debug_motor4_enabled = (fabsf(joy1_y) > MOTOR_ENABLE_THRESHOLD); // 수정

  debug_pwm1_duty = __HAL_TIM_GET_COMPARE(&htim3, TIM_CHANNEL_1);
  debug_pwm2_duty = __HAL_TIM_GET_COMPARE(&htim3, TIM_CHANNEL_2);
  debug_pwm3_duty = __HAL_TIM_GET_COMPARE(&htim3, TIM_CHANNEL_3);
  debug_pwm4_duty = __HAL_TIM_GET_COMPARE(&htim3, TIM_CHANNEL_4);

  // Z축 제어 로직은 함수 상단으로 이동됨
}

/* 젯슨으로 데이터 패킷 전송 (14 bytes) */
void UART_Send_Packet(void) {
  uint8_t packet[UART_TX_BUF_SIZE];

  // 1. Header
  packet[0] = PROTOCOL_HEADER1;
  packet[1] = PROTOCOL_HEADER2;

  // 2. Encoder Step (Motor 1 - Left X)
  int32_t step_data = (int32_t)virtual_steps_accumulator[0];
  packet[2] = (uint8_t)(step_data & 0xFF);
  packet[3] = (uint8_t)((step_data >> 8) & 0xFF);
  packet[4] = (uint8_t)((step_data >> 16) & 0xFF);
  packet[5] = (uint8_t)((step_data >> 24) & 0xFF);

  // 3. Action Logging (Normalized -100 ~ 100 char)
  packet[6] = (int8_t)(joy1_x * 100.0f);
  packet[7] = (int8_t)(joy1_y * 100.0f);
  packet[8] = (int8_t)(joy2_x * 100.0f);
  packet[9] = (int8_t)(joy2_y * 100.0f);

  // Winch Status (Simple 0/1/2)
  if (z_up_pressed)
    packet[10] = 1;
  else if (z_down_pressed)
    packet[10] = 2;
  else
    packet[10] = 0;

  // 4. Mode
  packet[11] = is_auto_mode ? 0x01 : 0x00;

  // 5. Status
  packet[12] = emergency_stop_active ? 0x02 : 0x01;

  // 6. Checksum (Sum of bytes 2-12)
  uint8_t checksum = 0;
  for (int i = 2; i <= 12; i++) {
    checksum += packet[i];
  }
  packet[13] = checksum;

  // Transmit
  uart1_tx(packet, UART_TX_BUF_SIZE);
}

// UART RX Complete Callback
void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart) {
  if (huart->Instance == USART1) {
    if (uart_rx_buffer[0] == PROTOCOL_HEADER1 &&
        uart_rx_buffer[1] == PROTOCOL_HEADER2) {
      Parse_UART_Packet();
      last_uart_rx_time = HAL_GetTick();
    }
    HAL_UART_Receive_IT(&huart1, uart_rx_buffer, UART_RX_BUF_SIZE);
  }
}

// v1.3: Parse Packet & Update Auto Commands
void Parse_UART_Packet(void) {
  uint8_t calc_sum = 0;
  for (int i = 2; i < 7; i++) {
    calc_sum += uart_rx_buffer[i];
  }

  if (calc_sum != uart_rx_buffer[7])
    return; // Checksum Fail

  is_auto_mode = true;
  last_uart_rx_time = HAL_GetTick();

  auto_cmd_x1 = ((int8_t)uart_rx_buffer[2]) / 100.0f;
  auto_cmd_y1 = ((int8_t)uart_rx_buffer[3]) / 100.0f;
  auto_cmd_x2 = ((int8_t)uart_rx_buffer[4]) / 100.0f;
  auto_cmd_y2 = ((int8_t)uart_rx_buffer[5]) / 100.0f;

  int8_t winch_val = (int8_t)uart_rx_buffer[6];
  if (winch_val > 10) {
    auto_cmd_winch_in = true;
    auto_cmd_winch_out = false;
  } else if (winch_val < -10) {
    auto_cmd_winch_in = false;
    auto_cmd_winch_out = true;
  } else {
    auto_cmd_winch_in = false;
    auto_cmd_winch_out = false;
  }
}

static void uart1_tx(const uint8_t *data, uint16_t len) {
  (void)HAL_UART_Transmit(&huart1, (uint8_t *)data, len, 100);
}

/* 디버거 대기 */
#if defined(DEBUG) || defined(_DEBUG)
static void debug_wait_for_attach(uint32_t max_wait_ms) {
  uint32_t t0 = HAL_GetTick();
  while ((HAL_GetTick() - t0) < max_wait_ms) {
    if (CoreDebug->DHCSR & CoreDebug_DHCSR_C_DEBUGEN_Msk) {
      break;
    }
  }
}
#else
static void debug_wait_for_attach(uint32_t max_wait_ms) { (void)max_wait_ms; }
#endif

/* USER CODE END 0 */

/**
 * @brief  The application entry point.
 * @retval int
 */
int main(void) {

  /* USER CODE BEGIN 1 */

  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick.
   */
  HAL_Init();

  /* USER CODE BEGIN Init */
  debug_wait_for_attach(500);
  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_DMA_Init();
  MX_USART1_UART_Init();
  MX_TIM3_Init();
  MX_ADC1_Init();
  /* USER CODE BEGIN 2 */

  /* 조이스틱 초기화 */
  Joystick_Init();

  /* TIM3 PWM 시작 */
  HAL_TIM_PWM_Start(&htim3, TIM_CHANNEL_1);
  HAL_TIM_PWM_Start(&htim3, TIM_CHANNEL_2);
  HAL_TIM_PWM_Start(&htim3, TIM_CHANNEL_3);
  HAL_TIM_PWM_Start(&htim3, TIM_CHANNEL_4);

  /* USER CODE END 2 */

  // UART RX Start
  HAL_UART_Receive_IT(&huart1, uart_rx_buffer, UART_RX_BUF_SIZE);

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  uint32_t last_control_time = HAL_GetTick();

  while (1) {
    uint32_t now = HAL_GetTick();

    // 1. 비상 정지 체크 (최우선)
    Emergency_Stop_Check();

    // 2. 제어 주기 (50ms)
    if ((now - last_control_time) >= CONTROL_PERIOD_MS) {
      last_control_time = now;

      // 조이스틱 읽기 (로깅용)
      Joystick_Read();

      // Hybrid Control Mode Switch
      if (is_auto_mode) {
        if ((now - last_uart_rx_time) > UART_TIMEOUT_MS) {
          is_auto_mode = false;
        } else {
          joy1_x = auto_cmd_x1;
          joy1_y = auto_cmd_y1;
          joy2_x = auto_cmd_x2;
          joy2_y = auto_cmd_y2;
          z_up_pressed = auto_cmd_winch_in;
          z_down_pressed = auto_cmd_winch_out;
        }
      }

      // 모터 제어
      Motor_Control();

      // 데이터 송신
      UART_Send_Packet();
    }

#if (MAIN_USE_WFI == 1)
    __WFI();
#endif
  }
  /* USER CODE END WHILE */

  /* USER CODE BEGIN 3 */
  /* USER CODE END 3 */
}

/**
 * @brief System Clock Configuration
 * @retval None
 */
void SystemClock_Config(void) {
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Configure the main internal regulator output voltage
   */
  __HAL_RCC_PWR_CLK_ENABLE();
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);

  /** Initializes the RCC Oscillators according to the specified parameters
   * in the RCC_OscInitTypeDef structure.
   */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSI;
  RCC_OscInitStruct.PLL.PLLM = 8;
  RCC_OscInitStruct.PLL.PLLN = 168;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV2;
  RCC_OscInitStruct.PLL.PLLQ = 4;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK) {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
   */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK | RCC_CLOCKTYPE_SYSCLK |
                                RCC_CLOCKTYPE_PCLK1 | RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV4;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV2;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_5) != HAL_OK) {
    Error_Handler();
  }
}

/**
 * @brief ADC1 Initialization Function
 * @param None
 * @retval None
 */
static void MX_ADC1_Init(void) {

  /* USER CODE BEGIN ADC1_Init 0 */

  /* USER CODE END ADC1_Init 0 */

  ADC_ChannelConfTypeDef sConfig = {0};

  /* USER CODE BEGIN ADC1_Init 1 */

  /* USER CODE END ADC1_Init 1 */

  /** Configure the global features of the ADC (Clock, Resolution, Data
   * Alignment and number of conversion)
   */
  hadc1.Instance = ADC1;
  hadc1.Init.ClockPrescaler = ADC_CLOCK_SYNC_PCLK_DIV4;
  hadc1.Init.Resolution = ADC_RESOLUTION_12B;
  hadc1.Init.ScanConvMode = ENABLE;
  hadc1.Init.ContinuousConvMode = ENABLE;
  hadc1.Init.DiscontinuousConvMode = DISABLE;
  hadc1.Init.ExternalTrigConvEdge = ADC_EXTERNALTRIGCONVEDGE_NONE;
  hadc1.Init.ExternalTrigConv = ADC_SOFTWARE_START;
  hadc1.Init.DataAlign = ADC_DATAALIGN_RIGHT;
  hadc1.Init.NbrOfConversion = 4;
  hadc1.Init.DMAContinuousRequests = ENABLE;
  hadc1.Init.EOCSelection = ADC_EOC_SINGLE_CONV;
  if (HAL_ADC_Init(&hadc1) != HAL_OK) {
    Error_Handler();
  }

  /** Configure for the selected ADC regular channel its corresponding rank in
   * the sequencer and its sample time.
   */
  sConfig.Channel = ADC_CHANNEL_10;
  sConfig.Rank = 1;
  sConfig.SamplingTime = ADC_SAMPLETIME_480CYCLES;
  if (HAL_ADC_ConfigChannel(&hadc1, &sConfig) != HAL_OK) {
    Error_Handler();
  }

  /** Configure for the selected ADC regular channel its corresponding rank in
   * the sequencer and its sample time.
   */
  sConfig.Channel = ADC_CHANNEL_11;
  sConfig.Rank = 2;
  if (HAL_ADC_ConfigChannel(&hadc1, &sConfig) != HAL_OK) {
    Error_Handler();
  }

  /** Configure for the selected ADC regular channel its corresponding rank in
   * the sequencer and its sample time.
   */
  sConfig.Channel = ADC_CHANNEL_12;
  sConfig.Rank = 3;
  if (HAL_ADC_ConfigChannel(&hadc1, &sConfig) != HAL_OK) {
    Error_Handler();
  }

  /** Configure for the selected ADC regular channel its corresponding rank in
   * the sequencer and its sample time.
   */
  sConfig.Channel = ADC_CHANNEL_13;
  sConfig.Rank = 4;
  if (HAL_ADC_ConfigChannel(&hadc1, &sConfig) != HAL_OK) {
    Error_Handler();
  }
  /* USER CODE BEGIN ADC1_Init 2 */

  /* USER CODE END ADC1_Init 2 */
}

/**
 * @brief TIM3 Initialization Function
 * @param None
 * @retval None
 */
static void MX_TIM3_Init(void) {

  /* USER CODE BEGIN TIM3_Init 0 */

  /* USER CODE END TIM3_Init 0 */

  TIM_ClockConfigTypeDef sClockSourceConfig = {0};
  TIM_MasterConfigTypeDef sMasterConfig = {0};
  TIM_OC_InitTypeDef sConfigOC = {0};

  /* USER CODE BEGIN TIM3_Init 1 */

  /* USER CODE END TIM3_Init 1 */
  htim3.Instance = TIM3;
  htim3.Init.Prescaler = 16;
  htim3.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim3.Init.Period = 999;
  htim3.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim3.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  if (HAL_TIM_Base_Init(&htim3) != HAL_OK) {
    Error_Handler();
  }
  sClockSourceConfig.ClockSource = TIM_CLOCKSOURCE_INTERNAL;
  if (HAL_TIM_ConfigClockSource(&htim3, &sClockSourceConfig) != HAL_OK) {
    Error_Handler();
  }
  if (HAL_TIM_PWM_Init(&htim3) != HAL_OK) {
    Error_Handler();
  }
  sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization(&htim3, &sMasterConfig) != HAL_OK) {
    Error_Handler();
  }
  sConfigOC.OCMode = TIM_OCMODE_PWM1;
  sConfigOC.Pulse = 500;
  sConfigOC.OCPolarity = TIM_OCPOLARITY_HIGH;
  sConfigOC.OCFastMode = TIM_OCFAST_DISABLE;
  if (HAL_TIM_PWM_ConfigChannel(&htim3, &sConfigOC, TIM_CHANNEL_1) != HAL_OK) {
    Error_Handler();
  }
  if (HAL_TIM_PWM_ConfigChannel(&htim3, &sConfigOC, TIM_CHANNEL_2) != HAL_OK) {
    Error_Handler();
  }
  if (HAL_TIM_PWM_ConfigChannel(&htim3, &sConfigOC, TIM_CHANNEL_3) != HAL_OK) {
    Error_Handler();
  }
  if (HAL_TIM_PWM_ConfigChannel(&htim3, &sConfigOC, TIM_CHANNEL_4) != HAL_OK) {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM3_Init 2 */

  /* USER CODE END TIM3_Init 2 */
  HAL_TIM_MspPostInit(&htim3);
}

/**
 * @brief USART1 Initialization Function
 * @param None
 * @retval None
 */
static void MX_USART1_UART_Init(void) {

  /* USER CODE BEGIN USART1_Init 0 */

  /* USER CODE END USART1_Init 0 */

  /* USER CODE BEGIN USART1_Init 1 */

  /* USER CODE END USART1_Init 1 */
  huart1.Instance = USART1;
  huart1.Init.BaudRate = 115200;
  huart1.Init.WordLength = UART_WORDLENGTH_8B;
  huart1.Init.StopBits = UART_STOPBITS_1;
  huart1.Init.Parity = UART_PARITY_NONE;
  huart1.Init.Mode = UART_MODE_TX_RX;
  huart1.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart1.Init.OverSampling = UART_OVERSAMPLING_16;
  if (HAL_UART_Init(&huart1) != HAL_OK) {
    Error_Handler();
  }
  /* USER CODE BEGIN USART1_Init 2 */

  /* USER CODE END USART1_Init 2 */
}

/**
 * Enable DMA controller clock
 */
static void MX_DMA_Init(void) {

  /* DMA controller clock enable */
  __HAL_RCC_DMA2_CLK_ENABLE();

  /* DMA interrupt init */
  /* DMA2_Stream0_IRQn interrupt configuration */
  HAL_NVIC_SetPriority(DMA2_Stream0_IRQn, 2, 0);
  HAL_NVIC_EnableIRQ(DMA2_Stream0_IRQn);
}

/**
 * @brief GPIO Initialization Function
 * @param None
 * @retval None
 */
static void MX_GPIO_Init(void) {
  GPIO_InitTypeDef GPIO_InitStruct = {0};
  /* USER CODE BEGIN MX_GPIO_Init_1 */

  /* USER CODE END MX_GPIO_Init_1 */

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOE_CLK_ENABLE();
  __HAL_RCC_GPIOC_CLK_ENABLE();
  __HAL_RCC_GPIOH_CLK_ENABLE();
  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_GPIOB_CLK_ENABLE();
  __HAL_RCC_GPIOD_CLK_ENABLE();

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(MAGNET_SW_GPIO_Port, MAGNET_SW_Pin, GPIO_PIN_RESET);

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(GPIOE, DIR_1_Pin | DIR_2_Pin | DIR_3_Pin | DIR_4_Pin,
                    GPIO_PIN_RESET);

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(GPIOE, ENA_1_Pin | ENA_2_Pin | ENA_3_Pin | ENA_4_Pin,
                    GPIO_PIN_SET);

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(GPIOD,
                    CYLINDER_EXTEND_Pin | CYLINDER_RETRACT_Pin | WINCH_OUT_Pin |
                        WINCH_IN_Pin,
                    GPIO_PIN_SET);

  /*Configure GPIO pins : Z_DOWN_Pin BTN_STOP_Pin Z_UPE1_Pin */
  GPIO_InitStruct.Pin = Z_DOWN_Pin | BTN_STOP_Pin | Z_UPE1_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
  GPIO_InitStruct.Pull = GPIO_PULLUP;
  HAL_GPIO_Init(GPIOE, &GPIO_InitStruct);

  /*Configure GPIO pin : MAGNET_SW_Pin */
  GPIO_InitStruct.Pin = MAGNET_SW_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(MAGNET_SW_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pins : ALM_X1_Pin ALM_X2_Pin ALM_Y1_Pin ALM_Y2_Pin */
  GPIO_InitStruct.Pin = ALM_X1_Pin | ALM_X2_Pin | ALM_Y1_Pin | ALM_Y2_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
  GPIO_InitStruct.Pull = GPIO_PULLUP;
  HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

  /*Configure GPIO pins : DIR_1_Pin DIR_2_Pin DIR_3_Pin DIR_4_Pin
                           ENA_1_Pin ENA_2_Pin ENA_3_Pin ENA_4_Pin */
  GPIO_InitStruct.Pin = DIR_1_Pin | DIR_2_Pin | DIR_3_Pin | DIR_4_Pin |
                        ENA_1_Pin | ENA_2_Pin | ENA_3_Pin | ENA_4_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOE, &GPIO_InitStruct);

  /*Configure GPIO pins : Z_UP_Pin Z_DOWNB11_Pin */
  GPIO_InitStruct.Pin = Z_UP_Pin | Z_DOWNB11_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
  GPIO_InitStruct.Pull = GPIO_PULLUP;
  HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

  /*Configure GPIO pins : CYLINDER_EXTEND_Pin CYLINDER_RETRACT_Pin WINCH_OUT_Pin
   * WINCH_IN_Pin */
  GPIO_InitStruct.Pin =
      CYLINDER_EXTEND_Pin | CYLINDER_RETRACT_Pin | WINCH_OUT_Pin | WINCH_IN_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOD, &GPIO_InitStruct);

  /*Configure GPIO pin : HOMING_Pin */
  GPIO_InitStruct.Pin = HOMING_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_IT_RISING;
  GPIO_InitStruct.Pull = GPIO_PULLUP;
  HAL_GPIO_Init(HOMING_GPIO_Port, &GPIO_InitStruct);

  /* USER CODE BEGIN MX_GPIO_Init_2 */

  /* USER CODE END MX_GPIO_Init_2 */
}

/* USER CODE BEGIN 4 */

/* USER CODE END 4 */

/**
 * @brief  This function is executed in case of error occurrence.
 * @retval None
 */
void Error_Handler(void) {
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1) {
  }
  /* USER CODE END Error_Handler_Debug */
}
#ifdef USE_FULL_ASSERT
/**
 * @brief  Reports the name of the source file and the source line number
 *         where the assert_param error has occurred.
 * @param  file: pointer to the source file name
 * @param  line: assert_param error line source number
 * @retval None
 */
void assert_failed(uint8_t *file, uint32_t line) {
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line
     number, ex: printf("Wrong parameters value: file %s on line %d\r\n", file,
     line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
