import asyncio


async def say(name: str) -> str:
    return f"Hello {name}"


async def main():
    try:
        f = say("World")
        f.send(None)
    except StopIteration as e:
        print(e.value)

    try:
        f = say("World").__await__()
        f.send(None)
    except StopIteration as e:
        print(e.value)


if __name__ == "__main__":
    asyncio.run(main())
