import math



def normalise(values, handedness,):

    hand = values.copy()

    if handedness == "Right":
        for x in range(0,63,3):
            hand[x] = 1 - hand[x]

    wrist_x =  hand[0]
    wrist_y = hand[1]
    wrist_z = hand[2]

    for x in range(0,63,3):
        hand[x] -= wrist_x

    for y in range(1,63,3):
        hand[y] -= wrist_y
    
    for x in range(2,63,3):
        hand[x] -= wrist_z


    l_nine = math.sqrt(hand[27]**2 + hand[28]**2 + hand[29]**2)

    if l_nine == 0:
        raise ValueError("Landmark 9 vector distance is 0")
    
    for i in range(0,63):
        hand[i] = hand[i] / l_nine


    return hand


if __name__ == "__main__":

    # Test 1: degenerate hand, should raise
    try:
        normalise([0.5] * 63, "Left")
        print("FAIL: expected ValueError")
    except ValueError:
        print("PASS: degenerate input raised")

    # Test 2: real hand from the dataset
    with open("dataset_kaggle.csv") as f:
        reader = csv.reader(f)
        next(reader)
        row = next(reader)

    raw = [float(v) for v in row[:63]]
    handed = row[63]
    out = normalise(raw, handed)

    print("wrist (want 0,0,0):", out[0], out[1], out[2])
    print("landmark 9:", out[27], out[28], out[29])
    print(
        "dist to 9 (want 1.0):",
        math.sqrt(out[27] ** 2 + out[28] ** 2 + out[29] ** 2),
    )