#pragma once

#include "main.h"              // CubeMX가 생성하는 *_Pin, *_GPIO_Port 사용
#include "stm32f4xx_hal.h"

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================
// 0) 공통: Active Level 규약
// ============================================================
#define BSP_LEVEL_LOW   GPIO_PIN_RESET
#define BSP_LEVEL_HIGH  GPIO_PIN_SET

// ============================================================
// 1) WINCH (Relay) - Active Low
//    ON  = LOW,  OFF = HIGH
// ============================================================
// CubeMX User Label이 정확히 WINCH_IN / WINCH_OUT 이어야 함
#ifndef WINCH_IN_Pin
#error "WINCH_IN_Pin is not defined. Check CubeMX user label: WINCH_IN"
#endif
#ifndef WINCH_OUT_Pin
#error "WINCH_OUT_Pin is not defined. Check CubeMX user label: WINCH_OUT"
#endif

#define BSP_WINCH_ON    BSP_LEVEL_LOW
#define BSP_WINCH_OFF   BSP_LEVEL_HIGH

#define BSP_WINCH_IN_PORT   WINCH_IN_GPIO_Port
#define BSP_WINCH_IN_PIN    WINCH_IN_Pin
#define BSP_WINCH_OUT_PORT  WINCH_OUT_GPIO_Port
#define BSP_WINCH_OUT_PIN   WINCH_OUT_Pin

static inline void BSP_WinchIn_On(void)   { HAL_GPIO_WritePin(BSP_WINCH_IN_PORT,  BSP_WINCH_IN_PIN,  BSP_WINCH_ON);  }
static inline void BSP_WinchIn_Off(void)  { HAL_GPIO_WritePin(BSP_WINCH_IN_PORT,  BSP_WINCH_IN_PIN,  BSP_WINCH_OFF); }
static inline void BSP_WinchOut_On(void)  { HAL_GPIO_WritePin(BSP_WINCH_OUT_PORT, BSP_WINCH_OUT_PIN, BSP_WINCH_ON);  }
static inline void BSP_WinchOut_Off(void) { HAL_GPIO_WritePin(BSP_WINCH_OUT_PORT, BSP_WINCH_OUT_PIN, BSP_WINCH_OFF); }

// ============================================================
// 2) STEPPER Driver ENA - Active Low
//    ENABLE = LOW, DISABLE = HIGH  (부팅 안전 기본값 = DISABLE)
// ============================================================
#ifndef ENA_1_Pin
#error "ENA_1_Pin is not defined. Check CubeMX user label: ENA_1"
#endif
#ifndef ENA_2_Pin
#error "ENA_2_Pin is not defined. Check CubeMX user label: ENA_2"
#endif
#ifndef ENA_3_Pin
#error "ENA_3_Pin is not defined. Check CubeMX user label: ENA_3"
#endif
#ifndef ENA_4_Pin
#error "ENA_4_Pin is not defined. Check CubeMX user label: ENA_4"
#endif

#define BSP_ENA_ENABLE   BSP_LEVEL_LOW
#define BSP_ENA_DISABLE  BSP_LEVEL_HIGH

#define BSP_ENA1_PORT  ENA_1_GPIO_Port
#define BSP_ENA1_PIN   ENA_1_Pin
#define BSP_ENA2_PORT  ENA_2_GPIO_Port
#define BSP_ENA2_PIN   ENA_2_Pin
#define BSP_ENA3_PORT  ENA_3_GPIO_Port
#define BSP_ENA3_PIN   ENA_3_Pin
#define BSP_ENA4_PORT  ENA_4_GPIO_Port
#define BSP_ENA4_PIN   ENA_4_Pin

static inline void BSP_Stepper1_Enable(void)  { HAL_GPIO_WritePin(BSP_ENA1_PORT, BSP_ENA1_PIN, BSP_ENA_ENABLE);  }
static inline void BSP_Stepper1_Disable(void) { HAL_GPIO_WritePin(BSP_ENA1_PORT, BSP_ENA1_PIN, BSP_ENA_DISABLE); }
static inline void BSP_Stepper2_Enable(void)  { HAL_GPIO_WritePin(BSP_ENA2_PORT, BSP_ENA2_PIN, BSP_ENA_ENABLE);  }
static inline void BSP_Stepper2_Disable(void) { HAL_GPIO_WritePin(BSP_ENA2_PORT, BSP_ENA2_PIN, BSP_ENA_DISABLE); }
static inline void BSP_Stepper3_Enable(void)  { HAL_GPIO_WritePin(BSP_ENA3_PORT, BSP_ENA3_PIN, BSP_ENA_ENABLE);  }
static inline void BSP_Stepper3_Disable(void) { HAL_GPIO_WritePin(BSP_ENA3_PORT, BSP_ENA3_PIN, BSP_ENA_DISABLE); }
static inline void BSP_Stepper4_Enable(void)  { HAL_GPIO_WritePin(BSP_ENA4_PORT, BSP_ENA4_PIN, BSP_ENA_ENABLE);  }
static inline void BSP_Stepper4_Disable(void) { HAL_GPIO_WritePin(BSP_ENA4_PORT, BSP_ENA4_PIN, BSP_ENA_DISABLE); }

// ============================================================
// 3) STEPPER DIR (방향) - 논리는 서비스에서 정의
// ============================================================
#ifndef DIR_1_Pin
#error "DIR_1_Pin is not defined. Check CubeMX user label: DIR_1"
#endif
#ifndef DIR_2_Pin
#error "DIR_2_Pin is not defined. Check CubeMX user label: DIR_2"
#endif
#ifndef DIR_3_Pin
#error "DIR_3_Pin is not defined. Check CubeMX user label: DIR_3"
#endif
#ifndef DIR_4_Pin
#error "DIR_4_Pin is not defined. Check CubeMX user label: DIR_4"
#endif

#define BSP_DIR1_PORT  DIR_1_GPIO_Port
#define BSP_DIR1_PIN   DIR_1_Pin
#define BSP_DIR2_PORT  DIR_2_GPIO_Port
#define BSP_DIR2_PIN   DIR_2_Pin
#define BSP_DIR3_PORT  DIR_3_GPIO_Port
#define BSP_DIR3_PIN   DIR_3_Pin
#define BSP_DIR4_PORT  DIR_4_GPIO_Port
#define BSP_DIR4_PIN   DIR_4_Pin

// ============================================================
// 4) ALM (Limit/Alarm) 입력 - Pull-up 기준, TRIGGERED = LOW 가정
// ============================================================
#ifndef ALM_X1_Pin
#error "ALM_X1_Pin is not defined. Check CubeMX user label: ALM_X1"
#endif
#ifndef ALM_X2_Pin
#error "ALM_X2_Pin is not defined. Check CubeMX user label: ALM_X2"
#endif
#ifndef ALM_Y1_Pin
#error "ALM_Y_Pin is not defined. Check CubeMX user label: ALM_Y"
#endif
#ifndef ALM_Y2_Pin
#error "ALM_Y_Pin is not defined. Check CubeMX user label: ALM_Y"
#endif


#define BSP_ALM_TRIGGERED  BSP_LEVEL_LOW
#define BSP_ALM_RELEASED   BSP_LEVEL_HIGH

#define BSP_ALM_X1_PORT  ALM_X1_GPIO_Port
#define BSP_ALM_X1_PIN   ALM_X1_Pin
#define BSP_ALM_X2_PORT  ALM_X2_GPIO_Port
#define BSP_ALM_X2_PIN   ALM_X2_Pin
#define BSP_ALM_Y1_PORT  ALM_Y1_GPIO_Port
#define BSP_ALM_Y1_PIN   ALM_Y1_Pin
#define BSP_ALM_Y2_PORT  ALM_Y2_GPIO_Port
#define BSP_ALM_Y2_PIN   ALM_Y2_Pin


static inline GPIO_PinState BSP_AlmX1_Read(void) { return HAL_GPIO_ReadPin(BSP_ALM_X1_PORT, BSP_ALM_X1_PIN); }
static inline GPIO_PinState BSP_AlmX2_Read(void) { return HAL_GPIO_ReadPin(BSP_ALM_X2_PORT, BSP_ALM_X2_PIN); }
static inline GPIO_PinState BSP_AlmY1_Read(void)  { return HAL_GPIO_ReadPin(BSP_ALM_Y1_PORT,  BSP_ALM_Y1_PIN);  }
static inline GPIO_PinState BSP_AlmY2_Read(void)  { return HAL_GPIO_ReadPin(BSP_ALM_Y2_PORT,  BSP_ALM_Y2_PIN);  }

static inline uint8_t BSP_AlmX1_IsTriggered(void) { return (BSP_AlmX1_Read() == BSP_ALM_TRIGGERED); }
static inline uint8_t BSP_AlmX2_IsTriggered(void) { return (BSP_AlmX2_Read() == BSP_ALM_TRIGGERED); }
static inline uint8_t BSP_AlmY1_IsTriggered(void)  { return (BSP_AlmY1_Read()  == BSP_ALM_TRIGGERED); }
static inline uint8_t BSP_AlmY2_IsTriggered(void)  { return (BSP_AlmY2_Read()  == BSP_ALM_TRIGGERED); }

// ============================================================
// 5) TIM3 PWM 채널 매핑 (현재 3축 기준)
// ============================================================
#define BSP_PWM_TIM           TIM3
#define BSP_PWM_CH_X1         TIM_CHANNEL_1
#define BSP_PWM_CH_X2         TIM_CHANNEL_2
#define BSP_PWM_CH_Y1          TIM_CHANNEL_3
#define BSP_PWM_CH_Y2          TIM_CHANNEL_4

// ============================================================
// 6) LEGACY: MAGNET_SW는 더 이상 사용하지 않음 (명시만)
// ============================================================
// 필요하면 CubeMX에서 핀 자체를 Free로 만들거나, 코드에서 완전 제거는 이후 단계에서.
#ifdef MAGNET_SW_Pin
#define BSP_MAGNET_LEGACY_PORT  MAGNET_SW_GPIO_Port
#define BSP_MAGNET_LEGACY_PIN   MAGNET_SW_Pin
#endif

#ifdef __cplusplus
}
#endif
