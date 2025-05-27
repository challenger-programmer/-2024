using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;
using System.Linq;
using UnityEditor.Rendering;


public class FishSelect : MonoBehaviour
{
    public TMP_Dropdown dropdown;
    public List<Quaternion> rotations;
    public List<Vector3> scales;
    private int fishNumber=0;
    public GameObject[] fishes;
    public List<Vector3> positions;
    private int roomNumber;
    private List<string> fishNames = new List<string>() {"魚1","魚2","魚3"};

    [System.Serializable]
    public class FishNamesWrapper
    {
        public string[] fishNames;  // 配列をラップするクラス
    }

    public void FishChange()
    {
        string selectedText = dropdown.captionText.text;
        HandleSelection(selectedText);
    }
    private void HandleSelection(string selectedText)
    {
        // シーン内のすべてのオブジェクトを取得
        GameObject[] allObjects = FindObjectsOfType<GameObject>();

        // 指定した名前のオブジェクトを探して削除
        foreach (GameObject obj in allObjects)
        {
            if (obj.name == $"FishV{fishNumber + 1}-2(Clone)")
            {
                Destroy(obj);
            }
        }

        if (selectedText == fishNames[0])
        {
            fishNumber = 0;
        }
        else if (selectedText == fishNames[1])
        {
            fishNumber = 1;
        }
        else if (selectedText == fishNames[2])
        {
            fishNumber = 2;
        }

        Quaternion rotation = rotations[fishNumber];
        Vector3 scale = scales[fishNumber];
        GameObject newFish1 = Instantiate(fishes[fishNumber], positions[0], rotation);
        GameObject newFish2 = Instantiate(fishes[fishNumber], positions[1], rotation);
        GameObject newFish3 = Instantiate(fishes[fishNumber], positions[2], rotation);
        newFish1.transform.localScale = scale;
        newFish2.transform.localScale = scale;
        newFish3.transform.localScale = scale;
    }

    //jsとのデータの受け渡し
    public void ReceiveFishDataFromJS(string fishNamesJson)
    {
        FishNamesWrapper fishNamesWrapper = JsonUtility.FromJson<FishNamesWrapper>(fishNamesJson);
        fishNames = fishNamesWrapper.fishNames.ToList();  // デシリアライズしてリストに変換
        for (int i = 0; i < 3; i++)
        {
            dropdown.options[i].text = fishNames[i];
        }
        dropdown.RefreshShownValue();
    }
}