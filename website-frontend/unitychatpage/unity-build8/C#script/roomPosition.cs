using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class roomPosition : MonoBehaviour
{
    public int room;
    public GameObject targetObject;
    // Start is called before the first frame update
    void Start()
    {
        room--;
        List<Vector3> position = new List<Vector3>
        {
            new Vector3(0f, 0f, 0f),
            new Vector3(3.87f, -10.03f, -49.64f),
        };
        targetObject.transform.position = position[room];
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
