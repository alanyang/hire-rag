from typing import List
from pydantic import BaseModel, Field
from toolz.curried import pipe, map, filter
from toolz.dicttoolz import get_in


class UserContactModel(BaseModel):
    phone: str | None = Field(None, max_length=15)
    email: str | None = Field(None, max_length=50)


class UserModel(BaseModel):
    name: str = Field(..., max_length=50)
    contact: UserContactModel | None = None


user = UserModel(
    name="Alice",
    contact=UserContactModel(
        phone="123-456-7890",
        email="alice@example.com",
    ),
)

print(get_in(["contact", "phone"], user.model_dump(), default="N/A"))


ages = [3, 45, 23, 18, 56, 27, 32, 42, 19, 25]

t = pipe(ages, filter(lambda x: x > 12), map(lambda x: x + 1), list)
