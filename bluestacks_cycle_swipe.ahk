; ====================================
; Циклические свайпы для BlueStacks
; Нажимайте C для свайпов: влево → вниз → вправо → вверх
; ====================================

#SingleInstance Force
SetWorkingDir %A_ScriptDir%

; Глобальная переменная для отслеживания текущего шага
global currentStep := 0

; Расстояние свайпа (можно изменить)
global swipeDistance := 150

; Скорость свайпа (чем меньше, тем быстрее)
global swipeSpeed := 10

; Привязка к клавише C
c::
{
    ; Проверяем, что активно окно BlueStacks
    IfWinActive, ahk_exe HD-Player.exe
    {
        ; Получаем координаты окна BlueStacks
        WinGetPos, winX, winY, winWidth, winHeight, ahk_exe HD-Player.exe
        
        ; Вычисляем центр окна
        centerX := winX + (winWidth // 2)
        centerY := winY + (winHeight // 2)
        
        ; Выполняем свайп в зависимости от текущего шага
        if (currentStep = 0)
        {
            ; Свайп ВЛЕВО
            SwipeLeft(centerX, centerY)
            ToolTip, Свайп ВЛЕВО (1/4)
        }
        else if (currentStep = 1)
        {
            ; Свайп ВНИЗ
            SwipeDown(centerX, centerY)
            ToolTip, Свайп ВНИЗ (2/4)
        }
        else if (currentStep = 2)
        {
            ; Свайп ВПРАВО
            SwipeRight(centerX, centerY)
            ToolTip, Свайп ВПРАВО (3/4)
        }
        else if (currentStep = 3)
        {
            ; Свайп ВВЕРХ
            SwipeUp(centerX, centerY)
            ToolTip, Свайп ВВЕРХ (4/4)
        }
        
        ; Переходим к следующему шагу (циклично 0-3)
        currentStep := Mod(currentStep + 1, 4)
        
        ; Убираем подсказку через 1 секунду
        SetTimer, RemoveToolTip, 1000
    }
    return
}

; Функция для свайпа влево
SwipeLeft(x, y)
{
    global swipeDistance, swipeSpeed
    MouseMove, x, y, 0
    Click down
    Sleep, 50
    MouseMove, x - swipeDistance, y, swipeSpeed
    Sleep, 50
    Click up
}

; Функция для свайпа вниз
SwipeDown(x, y)
{
    global swipeDistance, swipeSpeed
    MouseMove, x, y, 0
    Click down
    Sleep, 50
    MouseMove, x, y + swipeDistance, swipeSpeed
    Sleep, 50
    Click up
}

; Функция для свайпа вправо
SwipeRight(x, y)
{
    global swipeDistance, swipeSpeed
    MouseMove, x, y, 0
    Click down
    Sleep, 50
    MouseMove, x + swipeDistance, y, swipeSpeed
    Sleep, 50
    Click up
}

; Функция для свайпа вверх
SwipeUp(x, y)
{
    global swipeDistance, swipeSpeed
    MouseMove, x, y, 0
    Click down
    Sleep, 50
    MouseMove, x, y - swipeDistance, swipeSpeed
    Sleep, 50
    Click up
}

; Убрать подсказку
RemoveToolTip:
SetTimer, RemoveToolTip, Off
ToolTip
return

; ESC для выхода из скрипта
Esc::ExitApp
