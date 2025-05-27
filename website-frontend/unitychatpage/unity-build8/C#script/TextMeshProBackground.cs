using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;

[RequireComponent(typeof(TextMeshPro))]
public class TextMeshProBackground : MonoBehaviour
{
    public float PaddingTop;
    public float PaddingBottom;
    public float PaddingLeft;
    public float PaddingRight;
    public Material material;

    private GameObject Background;
    private TextMeshPro textMeshPro;

    void Awake()
    {
        this.textMeshPro = GetComponent<TextMeshPro>();

        // Planeを生成し、背景として設定
        this.Background = GameObject.CreatePrimitive(PrimitiveType.Plane);
        this.Background.name = "background";

        // BackgroundをTextMeshProの子オブジェクトにする
        this.Background.transform.SetParent(this.transform);

        // Materialを設定
        if (material != null)
            this.Background.GetComponent<MeshRenderer>().material = material;

        // 初期位置の計算
        UpdateBackgroundTransform();
    }

    void Update()
    {
        UpdateBackgroundTransform();
    }

    void UpdateBackgroundTransform()
    {
        // TextMeshProのローカル回転を取得
        var textRotation = this.textMeshPro.transform.localRotation.eulerAngles;

        // Planeは初期設定でX軸に-90度回転しているため、X軸の回転を調整
        this.Background.transform.localRotation = Quaternion.Euler(textRotation.x + 90, textRotation.y + 90, textRotation.z);

        // Boundsを取得
        var bounds = this.textMeshPro.bounds;

        // 背景の位置の計算
        var pos = bounds.center;
        var hoseiX = -(PaddingLeft / 2) + (PaddingRight / 2);
        var hoseiY = -(PaddingBottom / 2) + (PaddingTop / 2);
        var hoseiZ = 0.01f;
        this.Background.transform.localPosition = new Vector3(pos.x + hoseiX, pos.y + hoseiY, pos.z + hoseiZ);

        // 背景のサイズの計算
        var scale = bounds.extents;
        var hoseiW = (PaddingLeft + PaddingRight) / 10;
        var hoseiH = (PaddingTop + PaddingBottom) / 10;
        this.Background.transform.localScale = new Vector3((scale.x / 10 * 2) + hoseiW, 1, (scale.y / 10 * 2) + hoseiH);
    }
}
