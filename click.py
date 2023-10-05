import mouse
import time

inp1 = input("cycles:")

print("waiting...")
time.sleep(5)
for x in range(int(inp1)):
    print(x)
    time.sleep(1)
    mouse.click()
    mouse.press(button='left')
    mouse.release(button='left')