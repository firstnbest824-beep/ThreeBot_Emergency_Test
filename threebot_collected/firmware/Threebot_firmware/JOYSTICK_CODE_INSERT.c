/* ============================================================================
 * 조이스틱 제어 시스템 - CubeMX USER CODE 삽입용
 * ============================================================================
 * 이 코드를 CubeMX 재생성 후 main.c의 USER CODE 섹션에 복사하세요.
 *
 * 필수 전제조건:
 * - .ioc 파일에서 ADC1 (PC0-PC3) 설정 완료
 * - .ioc 파일에서 TIM3_CH4 (PB1) 설정 완료
 * - .ioc 파일에서 GPIO Input (PB10, PB11) 설정 완료
 * ============================================================================
 */

// ============================================================================
// SECTION 1: USER CODE BEGIN PD (Private Defines)
// ============================================================================

// ADC 설정
#define ADC_CHANNELS 4
#define ADC_RESOLUTION 4095.0f
#define ADC_CENTER 2048.0f
#define ADC_DEADZONE 100

// 모터 제어
#define MOTOR_MAX_SPEED 500
#define SLEW_RATE_LIMIT 0.1f

// 타이밍
#define CONTROL_PERIOD_MS 50
#define DEBOUNCE_TIME_MS 10

// 핀 정의 (CubeMX 생성 헤더에서 가져옴)
extern GPIO_TypeDef *GPIOB;
extern GPIO_TypeDef *GPIOD;
extern GPIO_TypeDef *GPIOE;

// ============================================================================
// SECTION 2: USER CODE BEGIN PV (Private Variables)
// ============================================================================

// ADC DMA 버퍼
volatile uint16_t adc_buffer[ADC_CHANNELS] = {0};

// 조이스틱 정규화 값
float joy1_x = 0.0f, joy1_y = 0.0f;
float joy2_x = 0.0f, joy2_y = 0.0f;
float prev_joy1_x = 0.0f, prev_joy1_y = 0.0f;
float prev_joy2_x = 0.0f, prev_joy2_y = 0.0f;

// Z축 버튼 상태
uint32_t z_up_last_time = 0, z_down_last_time = 0;
bool z_up_pressed = false, z_down_pressed = false;

bool emergency_stop_active = false;

// ============================================================================
// SECTION 3: USER CODE BEGIN PFP (Private Function Prototypes)
// ============================================================================

void Joystick_Init(void);
void Joystick_Read(void);
float Joystick_Normalize(uint16_t raw_value);
float Joystick_ApplySlewRate(float current, float previous);
bool Button_Read_Debounced(GPIO_TypeDef *GPIOx, uint16_t GPIO_Pin,
                           uint32_t *last_time, bool *state);
void Motor_Control(void);
void Emergency_Stop_Check(void);
void Motor_SetSpeed(uint32_t tim_channel, float speed);
void Motor_SetDirection(GPIO_TypeDef *GPIOx, uint16_t GPIO_Pin, bool forward);
void Motor_Enable(GPIO_TypeDef *GPIOx, uint16_t GPIO_Pin, bool enable);

// ============================================================================
// SECTION 4: USER CODE BEGIN 0 (함수 구현)
// ============================================================================

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

/* 조이스틱 값 읽기 */
void Joystick_Read(void) {
  // ADC 값 정규화
  float joy1_x_raw = Joystick_Normalize(adc_buffer[0]);
  float joy1_y_raw = Joystick_Normalize(adc_buffer[1]);
  float joy2_x_raw = Joystick_Normalize(adc_buffer[2]);
  float joy2_y_raw = Joystick_Normalize(adc_buffer[3]);

  // Slew Rate 적용
  joy1_x = Joystick_ApplySlewRate(joy1_x_raw, prev_joy1_x);
  joy1_y = Joystick_ApplySlewRate(joy1_y_raw, prev_joy1_y);
  joy2_x = Joystick_ApplySlewRate(joy2_x_raw, prev_joy2_x);
  joy2_y = Joystick_ApplySlewRate(joy2_y_raw, prev_joy2_y);

  // 이전 값 저장
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

/* 모터 활성화 */
void Motor_Enable(GPIO_TypeDef *GPIOx, uint16_t GPIO_Pin, bool enable) {
  HAL_GPIO_WritePin(GPIOx, GPIO_Pin, enable ? GPIO_PIN_SET : GPIO_PIN_RESET);
}

/* 비상 정지 체크 */
void Emergency_Stop_Check(void) {
  // BTN_STOP (PE0) 읽기 - Active Low
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
  if (emergency_stop_active)
    return;

  // ========== 그리퍼 1 (조이스틱 1) ==========
  // X1 축 (모터 1)
  Motor_Enable(GPIOE, ENA_1_Pin, fabsf(joy1_x) > 0.01f);
  Motor_SetDirection(GPIOE, DIR_1_Pin, joy1_x > 0);
  Motor_SetSpeed(TIM_CHANNEL_1, joy1_x);

  // Y1 축 (모터 2)
  Motor_Enable(GPIOE, ENA_2_Pin, fabsf(joy1_y) > 0.01f);
  Motor_SetDirection(GPIOE, DIR_2_Pin, joy1_y > 0);
  Motor_SetSpeed(TIM_CHANNEL_2, joy1_y);

  // ========== 그리퍼 2 (조이스틱 2) ==========
  // X2 축 (모터 3)
  Motor_Enable(GPIOE, ENA_3_Pin, fabsf(joy2_x) > 0.01f);
  Motor_SetDirection(GPIOE, DIR_3_Pin, joy2_x > 0);
  Motor_SetSpeed(TIM_CHANNEL_3, joy2_x);

  // Y2 축 (모터 4)
  Motor_Enable(GPIOE, ENA_4_Pin, fabsf(joy2_y) > 0.01f);
  Motor_SetDirection(GPIOE, DIR_4_Pin, joy2_y > 0);
  Motor_SetSpeed(TIM_CHANNEL_4, joy2_y);

  // ========== Z축 (윈치) ==========
  bool z_up =
      Button_Read_Debounced(GPIOB, Z_UP_Pin, &z_up_last_time, &z_up_pressed);
  bool z_down = Button_Read_Debounced(GPIOB, Z_DOWN_Pin, &z_down_last_time,
                                      &z_down_pressed);

  if (z_up && !z_down) {
    // 윈치 OUT (Active Low)
    HAL_GPIO_WritePin(GPIOD, WINCH_OUT_Pin, GPIO_PIN_RESET);
    HAL_GPIO_WritePin(GPIOD, WINCH_IN_Pin, GPIO_PIN_SET);
  } else if (z_down && !z_up) {
    // 윈치 IN (Active Low)
    HAL_GPIO_WritePin(GPIOD, WINCH_OUT_Pin, GPIO_PIN_SET);
    HAL_GPIO_WritePin(GPIOD, WINCH_IN_Pin, GPIO_PIN_RESET);
  } else {
    // 윈치 정지
    HAL_GPIO_WritePin(GPIOD, WINCH_OUT_Pin, GPIO_PIN_SET);
    HAL_GPIO_WritePin(GPIOD, WINCH_IN_Pin, GPIO_PIN_SET);
  }
}

// ============================================================================
// SECTION 5: USER CODE BEGIN 2 (초기화 코드)
// ============================================================================

/* main() 함수 내부, 페리페럴 초기화 직후에 삽입 */
/*
  // 조이스틱 초기화
  Joystick_Init();

  // TIM3 PWM 시작
  HAL_TIM_PWM_Start(&htim3, TIM_CHANNEL_1);
  HAL_TIM_PWM_Start(&htim3, TIM_CHANNEL_2);
  HAL_TIM_PWM_Start(&htim3, TIM_CHANNEL_3);
  HAL_TIM_PWM_Start(&htim3, TIM_CHANNEL_4);
*/

// ============================================================================
// SECTION 6: USER CODE BEGIN WHILE (메인 루프)
// ============================================================================

/* main() 함수의 while(1) 내부에 삽입 */
/*
  uint32_t last_control_time = HAL_GetTick();

  while (1)
  {
    // USER CODE BEGIN WHILE

    uint32_t now = HAL_GetTick();

    // 1. 비상 정지 체크 (최우선)
    Emergency_Stop_Check();

    // 2. 제어 주기 (50ms)
    if ((now - last_control_time) >= CONTROL_PERIOD_MS)
    {
        last_control_time = now;

        // 조이스틱 읽기
        Joystick_Read();

        // 모터 제어
        Motor_Control();
    }

    // USER CODE END WHILE
  }
*/

// ============================================================================
// 끝
// ============================================================================
