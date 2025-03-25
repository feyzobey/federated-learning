/* USER CODE BEGIN Header */
/**
 ******************************************************************************
 * @file           : main.c
 * @brief          : Main program body
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
/* Includes ------------------------------------------------------------------*/
#include "main.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */

/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
UART_HandleTypeDef huart2;
DMA_HandleTypeDef hdma_usart2_tx;
DMA_HandleTypeDef hdma_usart2_rx;

/* USER CODE BEGIN PV */
uint8_t rx_buffer[256];
uint8_t tx_buffer[2048]; // Increased buffer size for complete JSON
char json_buffer[2048];  // Increased buffer size for complete JSON
/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_DMA_Init(void);
static void MX_USART2_UART_Init(void);
/* USER CODE BEGIN PFP */

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */
int _write(int file, char *ptr, int len)
{
    HAL_UART_Transmit_DMA(&huart2, (uint8_t *)ptr, len);
    return len;
}

void send_dummy_model_data(int client_id)
{
    // Generate random values for each layer with correct shapes
    // Using simple text format with comma-separated values
    // Format: client_id,layer_name,value1,value2,...\n

    // Create text string with random weights for each client
    int len = snprintf(json_buffer, sizeof(json_buffer),
                       "CLIENT:%d\n"
                       "conv1.weight:%d,%d,%d,%d,%d,%d,%d,%d,%d\n"
                       "conv1.bias:%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n"
                       "conv2.weight:%d,%d,%d,%d,%d,%d,%d,%d,%d\n"
                       "conv2.bias:%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n"
                       "fc1.weight:%d,%d,%d,%d,%d,%d,%d,%d\n"
                       "fc1.bias:%d,%d,%d,%d\n"
                       "fc2.weight:%d,%d,%d,%d\n"
                       "fc2.bias:%d,%d\nEND\n",
                       client_id,
                       // Random values for conv1.weight (32x3x3)
                       (client_id * 13) % 100, (client_id * 17) % 100, (client_id * 19) % 100,
                       (client_id * 23) % 100, (client_id * 29) % 100, (client_id * 31) % 100,
                       (client_id * 37) % 100, (client_id * 41) % 100, (client_id * 43) % 100,
                       // Random values for conv1.bias (32)
                       (client_id * 47) % 100, (client_id * 53) % 100, (client_id * 59) % 100,
                       (client_id * 61) % 100, (client_id * 67) % 100, (client_id * 71) % 100,
                       (client_id * 73) % 100, (client_id * 79) % 100, (client_id * 83) % 100,
                       (client_id * 89) % 100, (client_id * 97) % 100, (client_id * 101) % 100,
                       (client_id * 103) % 100, (client_id * 107) % 100, (client_id * 109) % 100,
                       (client_id * 113) % 100, (client_id * 127) % 100, (client_id * 131) % 100,
                       (client_id * 137) % 100, (client_id * 139) % 100, (client_id * 149) % 100,
                       (client_id * 151) % 100,
                       // Random values for conv2.weight (64x32x3)
                       (client_id * 157) % 100, (client_id * 163) % 100, (client_id * 167) % 100,
                       (client_id * 173) % 100, (client_id * 179) % 100, (client_id * 181) % 100,
                       (client_id * 191) % 100, (client_id * 193) % 100, (client_id * 197) % 100,
                       // Random values for conv2.bias (64)
                       (client_id * 199) % 100, (client_id * 211) % 100, (client_id * 223) % 100,
                       (client_id * 227) % 100, (client_id * 229) % 100, (client_id * 233) % 100,
                       (client_id * 239) % 100, (client_id * 241) % 100, (client_id * 251) % 100,
                       (client_id * 257) % 100, (client_id * 263) % 100, (client_id * 269) % 100,
                       (client_id * 271) % 100, (client_id * 277) % 100, (client_id * 281) % 100,
                       (client_id * 283) % 100, (client_id * 293) % 100, (client_id * 307) % 100,
                       (client_id * 311) % 100, (client_id * 313) % 100, (client_id * 317) % 100,
                       (client_id * 331) % 100, (client_id * 337) % 100, (client_id * 347) % 100,
                       (client_id * 349) % 100, (client_id * 353) % 100, (client_id * 359) % 100,
                       (client_id * 367) % 100, (client_id * 373) % 100, (client_id * 379) % 100,
                       (client_id * 383) % 100, (client_id * 389) % 100, (client_id * 397) % 100,
                       (client_id * 401) % 100, (client_id * 409) % 100, (client_id * 419) % 100,
                       (client_id * 421) % 100, (client_id * 431) % 100, (client_id * 433) % 100,
                       (client_id * 439) % 100, (client_id * 443) % 100, (client_id * 449) % 100,
                       (client_id * 457) % 100, (client_id * 461) % 100, (client_id * 463) % 100,
                       (client_id * 467) % 100, (client_id * 479) % 100, (client_id * 487) % 100,
                       (client_id * 491) % 100, (client_id * 499) % 100, (client_id * 503) % 100,
                       (client_id * 509) % 100, (client_id * 521) % 100, (client_id * 523) % 100,
                       (client_id * 541) % 100, (client_id * 547) % 100, (client_id * 557) % 100,
                       (client_id * 563) % 100, (client_id * 569) % 100, (client_id * 571) % 100,
                       (client_id * 577) % 100, (client_id * 587) % 100, (client_id * 593) % 100,
                       (client_id * 599) % 100, (client_id * 601) % 100, (client_id * 607) % 100,
                       // Random values for fc1.weight
                       (client_id * 613) % 100, (client_id * 617) % 100, (client_id * 619) % 100,
                       (client_id * 631) % 100, (client_id * 641) % 100, (client_id * 643) % 100,
                       (client_id * 647) % 100, (client_id * 653) % 100,
                       // Random values for fc1.bias
                       (client_id * 659) % 100, (client_id * 661) % 100,
                       (client_id * 673) % 100, (client_id * 677) % 100,
                       // Random values for fc2.weight
                       (client_id * 683) % 100, (client_id * 691) % 100,
                       (client_id * 701) % 100, (client_id * 709) % 100,
                       // Random values for fc2.bias
                       (client_id * 719) % 100, (client_id * 727) % 100);

    if (len >= sizeof(json_buffer))
    {
        // Buffer overflow occurred
        Error_Handler();
    }

    // Toggle LED to show transmission
    HAL_GPIO_TogglePin(LD2_GPIO_Port, LD2_Pin);

    // Send the data and wait for transmission to complete
    HAL_UART_Transmit(&huart2, (uint8_t *)json_buffer, len, 5000);

    // Wait for transmission to complete
    while (HAL_UART_GetState(&huart2) != HAL_UART_STATE_READY)
    {
        HAL_Delay(1);
    }

    HAL_Delay(100); // Small delay between sends
}

void send_all_client_updates(void)
{
    // Send 36 different client updates
    for (int i = 1; i <= 36; i++)
    {
        send_dummy_model_data(i);
        // LED will toggle for each transmission
    }

    // Keep LED on to indicate completion
    HAL_GPIO_WritePin(LD2_GPIO_Port, LD2_Pin, GPIO_PIN_SET);
}
/* USER CODE END 0 */

/**
 * @brief  The application entry point.
 * @retval int
 */
int main(void)
{

    /* USER CODE BEGIN 1 */

    /* USER CODE END 1 */

    /* MCU Configuration--------------------------------------------------------*/

    /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
    HAL_Init();

    /* USER CODE BEGIN Init */

    /* USER CODE END Init */

    /* Configure the system clock */
    SystemClock_Config();

    /* USER CODE BEGIN SysInit */

    /* USER CODE END SysInit */

    /* Initialize all configured peripherals */
    MX_GPIO_Init();
    MX_DMA_Init();
    MX_USART2_UART_Init();
    /* USER CODE BEGIN 2 */
    send_all_client_updates();
    /* USER CODE END 2 */

    /* Infinite loop */
    /* USER CODE BEGIN WHILE */
    while (1)
    {
        /* USER CODE END WHILE */

        /* USER CODE BEGIN 3 */
        HAL_Delay(1000);
    }
    /* USER CODE END 3 */
}

/**
 * @brief System Clock Configuration
 * @retval None
 */
void SystemClock_Config(void)
{
    RCC_OscInitTypeDef RCC_OscInitStruct = {0};
    RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

    /** Configure the main internal regulator output voltage
     */
    if (HAL_PWREx_ControlVoltageScaling(PWR_REGULATOR_VOLTAGE_SCALE1) != HAL_OK)
    {
        Error_Handler();
    }

    /** Initializes the RCC Oscillators according to the specified parameters
     * in the RCC_OscInitTypeDef structure.
     */
    RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
    RCC_OscInitStruct.HSIState = RCC_HSI_ON;
    RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
    RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
    RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSI;
    RCC_OscInitStruct.PLL.PLLM = 1;
    RCC_OscInitStruct.PLL.PLLN = 10;
    RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV7;
    RCC_OscInitStruct.PLL.PLLQ = RCC_PLLQ_DIV2;
    RCC_OscInitStruct.PLL.PLLR = RCC_PLLR_DIV2;
    if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
    {
        Error_Handler();
    }

    /** Initializes the CPU, AHB and APB buses clocks
     */
    RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK | RCC_CLOCKTYPE_SYSCLK | RCC_CLOCKTYPE_PCLK1 | RCC_CLOCKTYPE_PCLK2;
    RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
    RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
    RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV1;
    RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

    if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_4) != HAL_OK)
    {
        Error_Handler();
    }
}

/**
 * @brief USART2 Initialization Function
 * @param None
 * @retval None
 */
static void MX_USART2_UART_Init(void)
{

    /* USER CODE BEGIN USART2_Init 0 */

    /* USER CODE END USART2_Init 0 */

    /* USER CODE BEGIN USART2_Init 1 */

    /* USER CODE END USART2_Init 1 */
    huart2.Instance = USART2;
    huart2.Init.BaudRate = 115200;
    huart2.Init.WordLength = UART_WORDLENGTH_8B;
    huart2.Init.StopBits = UART_STOPBITS_1;
    huart2.Init.Parity = UART_PARITY_NONE;
    huart2.Init.Mode = UART_MODE_TX_RX;
    huart2.Init.HwFlowCtl = UART_HWCONTROL_NONE;
    huart2.Init.OverSampling = UART_OVERSAMPLING_16;
    huart2.Init.OneBitSampling = UART_ONE_BIT_SAMPLE_DISABLE;
    huart2.AdvancedInit.AdvFeatureInit = UART_ADVFEATURE_NO_INIT;
    if (HAL_UART_Init(&huart2) != HAL_OK)
    {
        Error_Handler();
    }
    /* USER CODE BEGIN USART2_Init 2 */

    /* USER CODE END USART2_Init 2 */
}

/**
 * Enable DMA controller clock
 */
static void MX_DMA_Init(void)
{

    /* DMA controller clock enable */
    __HAL_RCC_DMA1_CLK_ENABLE();

    /* DMA interrupt init */
    /* DMA1_Channel6_IRQn interrupt configuration */
    HAL_NVIC_SetPriority(DMA1_Channel6_IRQn, 0, 0);
    HAL_NVIC_EnableIRQ(DMA1_Channel6_IRQn);
    /* DMA1_Channel7_IRQn interrupt configuration */
    HAL_NVIC_SetPriority(DMA1_Channel7_IRQn, 0, 0);
    HAL_NVIC_EnableIRQ(DMA1_Channel7_IRQn);
}

/**
 * @brief GPIO Initialization Function
 * @param None
 * @retval None
 */
static void MX_GPIO_Init(void)
{
    GPIO_InitTypeDef GPIO_InitStruct = {0};
    /* USER CODE BEGIN MX_GPIO_Init_1 */
    /* USER CODE END MX_GPIO_Init_1 */

    /* GPIO Ports Clock Enable */
    __HAL_RCC_GPIOC_CLK_ENABLE();
    __HAL_RCC_GPIOH_CLK_ENABLE();
    __HAL_RCC_GPIOA_CLK_ENABLE();
    __HAL_RCC_GPIOB_CLK_ENABLE();

    /*Configure GPIO pin Output Level */
    HAL_GPIO_WritePin(LD2_GPIO_Port, LD2_Pin, GPIO_PIN_RESET);

    /*Configure GPIO pin : B1_Pin */
    GPIO_InitStruct.Pin = B1_Pin;
    GPIO_InitStruct.Mode = GPIO_MODE_IT_FALLING;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    HAL_GPIO_Init(B1_GPIO_Port, &GPIO_InitStruct);

    /*Configure GPIO pin : LD2_Pin */
    GPIO_InitStruct.Pin = LD2_Pin;
    GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
    HAL_GPIO_Init(LD2_GPIO_Port, &GPIO_InitStruct);

    /* USER CODE BEGIN MX_GPIO_Init_2 */
    /* USER CODE END MX_GPIO_Init_2 */
}

/* USER CODE BEGIN 4 */

/* USER CODE END 4 */

/**
 * @brief  This function is executed in case of error occurrence.
 * @retval None
 */
void Error_Handler(void)
{
    /* USER CODE BEGIN Error_Handler_Debug */
    /* User can add his own implementation to report the HAL error return state */
    __disable_irq();
    while (1)
    {
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
void assert_failed(uint8_t *file, uint32_t line)
{
    /* USER CODE BEGIN 6 */
    /* User can add his own implementation to report the file name and line number,
       ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
    /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
