package com.example.appcamera;

import androidx.activity.result.ActivityResult;
import androidx.activity.result.ActivityResultCallback;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import org.tensorflow.lite.Interpreter;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.List;
import android.Manifest;


public class MainActivity extends AppCompatActivity {
    private Bitmap bitmap;
    private ImageView imageView;
    private TextView resultTextView;
    private Interpreter tflite;
    private static final String MODEL_PATH = "model_unquant.tflite";
    private static final String LABELS_PATH = "labels.txt";
    private static final int CAMERA_PERMISSION_REQUEST_CODE = 1001; // You can choose any unique integer value
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        imageView = findViewById(R.id.imageView);
        resultTextView = findViewById(R.id.resultTextView);

        Button picBtn = findViewById(R.id.pic_btn);
        picBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if(checkCameraPermission()){
                openCamera();
            }else {
                    requestCameraPermission();
                }
            }
        });

        Button pickImageButton = findViewById(R.id.pickImageBtn);
        pickImageButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                openGallery();
            }
        });

        Button classifyBtn = findViewById(R.id.classify_btn);
        classifyBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                classifyImage();
            }
        });

        initializeActivityLaunchers();
    }
    private void initializeActivityLaunchers() {
        activityResultPicture = registerForActivityResult(
                new ActivityResultContracts.StartActivityForResult(),
                new ActivityResultCallback<ActivityResult>() {
                    @Override
                    public void onActivityResult(ActivityResult result) {
                        if (result.getResultCode() == RESULT_OK && result.getData() != null) {
                            Bundle extras = result.getData().getExtras();
                            bitmap = (Bitmap) extras.get("data");
                            if (imageView != null) {
                                imageView.setImageBitmap(bitmap);
                            }
                        }
                    }
                });
        activityResultGallery = registerForActivityResult(
                new ActivityResultContracts.StartActivityForResult(),
                new ActivityResultCallback<ActivityResult>() {
                    @Override
                    public void onActivityResult(ActivityResult result) {
                        if (result.getResultCode() == RESULT_OK && result.getData() != null) {
                            Uri imageUri = result.getData().getData();
                            try {
                                bitmap = MediaStore.Images.Media.getBitmap(getContentResolver(), imageUri);
                                if (imageView != null) {
                                    imageView.setImageBitmap(bitmap);
                                }
                            } catch (IOException e) {
                                e.printStackTrace();
                            }
                        }
                    }
                });
    }
    private void classifyImage() {
        if (bitmap != null && imageView != null) {
            try {
                // Initialize the TensorFlow Lite interpreter
                tflite = new Interpreter(loadModelFile());

                // Preprocess the image and run inference
                String label = classify(bitmap);

                // Display the result in the TextView
                String labelStr[] = label.toString().split(" ");
                resultTextView.setText(labelStr[1]);

            } catch (IOException e) {
                e.printStackTrace();
            }
        } else {
            Toast.makeText(this, "먼저 사진을 촬영하거나 선택하세요.", Toast.LENGTH_SHORT).show();
        }
    }

    private MappedByteBuffer loadModelFile() throws IOException {
        AssetFileDescriptor fileDescriptor = getAssets().openFd(MODEL_PATH);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    private String classify(Bitmap bitmap) {
        // Preprocess the image and run inference using your TensorFlow Lite model
        // Modify this part based on your specific model and preprocessing requirements

        int imageTensorIndex = 0;
        int[] inputShape = tflite.getInputTensor(imageTensorIndex).shape();
        int imageSizeX = inputShape[1];
        int imageSizeY = inputShape[2];

        // Preprocess the bitmap into the required input size
        Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, imageSizeX, imageSizeY, true);
        ByteBuffer inputBuffer = convertBitmapToByteBuffer(resizedBitmap);

        // Run inference
        float[][] outputScores = new float[1][13];  // Modify NUM_CLASSES based on your model
        tflite.run(inputBuffer, outputScores);

        // Postprocess the result and return the predicted label
        String predictedLabel = postprocessResult(outputScores);

        return predictedLabel;
    }

    private ByteBuffer convertBitmapToByteBuffer(Bitmap bitmap) {
        int imageSizeX = tflite.getInputTensor(0).shape()[1];
        int imageSizeY = tflite.getInputTensor(0).shape()[2];
        int channels = tflite.getInputTensor(0).shape()[3];

        ByteBuffer inputBuffer = ByteBuffer.allocateDirect(4 * imageSizeX * imageSizeY * channels);
        inputBuffer.order(ByteOrder.nativeOrder());

        int[] pixels = new int[imageSizeX * imageSizeY];
        bitmap.getPixels(pixels, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

        int pixel = 0;
        for (int i = 0; i < imageSizeX; ++i) {
            for (int j = 0; j < imageSizeY; ++j) {
                final int val = pixels[pixel++];
                inputBuffer.putFloat(((val >> 16) & 0xFF) / 255.0f);
                inputBuffer.putFloat(((val >> 8) & 0xFF) / 255.0f);
                inputBuffer.putFloat((val & 0xFF) / 255.0f);
            }
        }

        return inputBuffer;
    }

    private String postprocessResult(float[][] outputScores) {
        // Postprocess the model output to get the predicted label
        // Modify this part based on your specific model and postprocessing requirements
        // In this example, it assumes the output is a single label with the highest probability

        int maxIndex = 0;
        float maxScore = outputScores[0][0];

        for (int i = 1; i < outputScores[0].length; i++) {
            if (outputScores[0][i] > maxScore) {
                maxScore = outputScores[0][i];
                maxIndex = i;
            }
        }

        List<String> labels = getLabels();  // Load labels from your labels.txt file
        return labels.get(maxIndex);
    }

    private List<String> getLabels() {
        List<String> labels = new ArrayList<>();

        try {
            InputStream is = getAssets().open(LABELS_PATH);
            BufferedReader reader = new BufferedReader(new InputStreamReader(is));

            String line;
            while ((line = reader.readLine()) != null) {
                labels.add(line);
            }

            reader.close();
        } catch (IOException e) {
            e.printStackTrace();
        }

        return labels;
    }

    private void openCamera() {
        Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        activityResultPicture.launch(intent);
    }

    private void openGallery() {
        Intent intent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
        activityResultGallery.launch(intent);
    }


    private boolean checkCameraPermission() {
        // Check if the camera permission is granted
        return ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED;
    }

    private void requestCameraPermission() {
        // Request camera permission
        ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, CAMERA_PERMISSION_REQUEST_CODE);
    }


    ActivityResultLauncher<Intent> activityResultPicture = registerForActivityResult(
            new ActivityResultContracts.StartActivityForResult(),
            new ActivityResultCallback<ActivityResult>() {
                @Override
                public void onActivityResult(ActivityResult result) {
                    if (result.getResultCode() == RESULT_OK && result.getData() != null) {
                        Bundle extras = result.getData().getExtras();
                        bitmap = (Bitmap) extras.get("data");
                        imageView.setImageBitmap(bitmap);
                    }
                }
            });

    ActivityResultLauncher<Intent> activityResultGallery = registerForActivityResult(
            new ActivityResultContracts.StartActivityForResult(),
            new ActivityResultCallback<ActivityResult>() {
                @Override
                public void onActivityResult(ActivityResult result) {
                    if (result.getResultCode() == RESULT_OK && result.getData() != null) {
                        Uri imageUri = result.getData().getData();
                        try {
                            bitmap = MediaStore.Images.Media.getBitmap(getContentResolver(), imageUri);
                            imageView.setImageBitmap(bitmap);
                        } catch (IOException e) {
                            e.printStackTrace();
                        }
                    }
                }
            });
}