using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UIElements;

public class FishReplace : MonoBehaviour
{
    // インスペクタでfishV1とfishV2のPrefabを指定
    public List<Quaternion> rotations;
    public List<Vector3> scales;
    public int fishNumber;
    public GameObject[] fishes;

    void Start()
    {
        fishNumber--;
        // fishV1のオーバーライド（位置、回転、スケール）を保存
        Vector3 position = this.gameObject.transform.position;
        Quaternion rotation = rotations[fishNumber];
        Vector3 scale = scales[fishNumber];

        Destroy(this.gameObject);

        // fishV2を同じ位置、回転、スケールでインスタンス化
        GameObject newFish = Instantiate(fishes[fishNumber], position, rotation);
        newFish.transform.localScale = scale;
    }
}

