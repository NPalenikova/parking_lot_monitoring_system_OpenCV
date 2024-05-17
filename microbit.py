from microbit import *

number = None
last_data_time = running_time()

while True:
    data = uart.read()
    
    if data:
        last_data_time = running_time()  # Update last data time
        if int(data) != number:
            number = int(data)
    
    # Check if no data received for 20 seconds
    if running_time() - last_data_time >= 20000:
        number = None
        display.clear()  # Clear the display if no data for 20 seconds
    
    if number:
        display.scroll(str(number))
