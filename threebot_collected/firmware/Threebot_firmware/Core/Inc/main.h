/* USER CODE BEGIN Header */
/**
 ******************************************************************************
 * @file           : main.h
 * @brief          : Header for main.c file.
 *                   This file contains the common defines of the application.
 ******************************************************************************
 * @attention
 *
 * Copyright (c) 2025 STMicroelectronics.
 * All rights reserved.
 *
 * This software is licensed under terms that can be found in the LICENSE file
 * in the root directory of this software component.
 * If no LICENSE file comes with this software, it is provided AS-IS.
 *
 ******************************************************************************
 */
/* USER CODE END Header */

/* Define to prevent recursive inclusion -------------------------------------*/
#ifndef __MAIN_H
#define __MAIN_H

#ifdef __cplusplus
extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#include "stm32f4xx_hal.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */

/* USER CODE END Includes */

/* Exported types ------------------------------------------------------------*/
/* USER CODE BEGIN ET */

/* USER CODE END ET */

/* Exported constants --------------------------------------------------------*/
/* USER CODE BEGIN EC */

/* USER CODE END EC */

/* Exported macro ------------------------------------------------------------*/
/* USER CODE BEGIN EM */

/* USER CODE END EM */

void HAL_TIM_MspPostInit(TIM_HandleTypeDef *htim);

/* Exported functions prototypes ---------------------------------------------*/
void Error_Handler(void);

/* USER CODE BEGIN EFP */

/* USER CODE END EFP */

/* Private defines -----------------------------------------------------------*/
#define Z_DOWN_Pin GPIO_PIN_2
#define Z_DOWN_GPIO_Port GPIOE
#define BTN_STOP_Pin GPIO_PIN_1
#define BTN_STOP_GPIO_Port GPIOE
#define MAGNET_SW_Pin GPIO_PIN_13
#define MAGNET_SW_GPIO_Port GPIOC
#define ALM_X1_Pin GPIO_PIN_0
#define ALM_X1_GPIO_Port GPIOA
#define ALM_X2_Pin GPIO_PIN_1
#define ALM_X2_GPIO_Port GPIOA
#define ALM_Y1_Pin GPIO_PIN_2
#define ALM_Y1_GPIO_Port GPIOA
#define ALM_Y2_Pin GPIO_PIN_3
#define ALM_Y2_GPIO_Port GPIOA
#define DIR_1_Pin GPIO_PIN_8
#define DIR_1_GPIO_Port GPIOE
#define DIR_2_Pin GPIO_PIN_9
#define DIR_2_GPIO_Port GPIOE
#define DIR_3_Pin GPIO_PIN_10
#define DIR_3_GPIO_Port GPIOE
#define DIR_4_Pin GPIO_PIN_11
#define DIR_4_GPIO_Port GPIOE
#define ENA_1_Pin GPIO_PIN_12
#define ENA_1_GPIO_Port GPIOE
#define ENA_2_Pin GPIO_PIN_13
#define ENA_2_GPIO_Port GPIOE
#define ENA_3_Pin GPIO_PIN_14
#define ENA_3_GPIO_Port GPIOE
#define ENA_4_Pin GPIO_PIN_15
#define ENA_4_GPIO_Port GPIOE
#define Z_UP_Pin GPIO_PIN_10
#define Z_UP_GPIO_Port GPIOB
#define Z_DOWNB11_Pin GPIO_PIN_11
#define Z_DOWNB11_GPIO_Port GPIOB
#define CYLINDER_EXTEND_Pin GPIO_PIN_10
#define CYLINDER_EXTEND_GPIO_Port GPIOD
#define CYLINDER_RETRACT_Pin GPIO_PIN_11
#define CYLINDER_RETRACT_GPIO_Port GPIOD
#define WINCH_OUT_Pin GPIO_PIN_12
#define WINCH_OUT_GPIO_Port GPIOD
#define WINCH_IN_Pin GPIO_PIN_13
#define WINCH_IN_GPIO_Port GPIOD
#define HOMING_Pin GPIO_PIN_0
#define HOMING_GPIO_Port GPIOE
#define Z_UPE1_Pin GPIO_PIN_3
#define Z_UPE1_GPIO_Port GPIOE

/* USER CODE BEGIN Private defines */

/* USER CODE END Private defines */

#ifdef __cplusplus
}
#endif

#endif /* __MAIN_H */
