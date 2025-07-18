import asyncio
from bleak import BleakScanner, BleakClient

async def dismain():
    devices = await BleakScanner.discover()
    print("found devices {}".format(len(devices)))
    for d in devices:
        print(d)



IMU_ADDRESS = 'E5:2C:A7:50:A0:97'
CHARACTERISTIC_UUID = "0000ffe4-0000-1000-8000-00805f9a34fb" # BLE characteristic to read data from
async def main():
    print(f"Connecting to IMU at {IMU_ADDRESS}")
    try:
        async with BleakClient(IMU_ADDRESS) as client:
            print("Connected. Start doing reps when ready.")
    except Exception as e:
        print(f"BLE connection error: {e}")

asyncio.run(dismain())
asyncio.run(main())