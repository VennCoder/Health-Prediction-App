import React, { useState } from 'react';
import { View, Text, TextInput, Button, StyleSheet, Switch, ScrollView, Alert } from 'react-native';
import axios from 'axios';

export default function App() {
  const [predictionType, setPredictionType] = useState('disease'); // 'disease' or 'stress'
  const [symptoms, setSymptoms] = useState('');
  const [stressInputs, setStressInputs] = useState({
    X: '',
    Y: '',
    Z: '',
    EDA: '',
    HR: '',
    TEMP: '',
  });
  const [response, setResponse] = useState(null);

  const togglePredictionType = () => {
    setPredictionType((prev) => (prev === 'disease' ? 'stress' : 'disease'));
    setResponse(null); // Clear response on toggle
  };

  const handleStressInputChange = (key, value) => {
    // Allow only numeric values, including negatives and a single dot for decimals
    if (/^-?\d*\.?\d*$/.test(value)) {
      setStressInputs({ ...stressInputs, [key]: value });
    }
  };

  const submitPrediction = async () => {
    try {
      let payload;
      if (predictionType === 'disease') {
        // Convert symptoms string to array or pass natural language
        if (/^[01]+$/.test(symptoms)) {
          payload = {
            prediction_type: 'disease',
            symptoms: symptoms.split('').map(Number),
          };
        } else {
          payload = {
            prediction_type: 'disease',
            symptoms,
          };
        }
      } else {
        // Create stress prediction payload
        const { X, Y, Z, EDA, HR, TEMP } = stressInputs;
        payload = {
          prediction_type: 'stress',
          X: parseFloat(X),
          Y: parseFloat(Y),
          Z: parseFloat(Z),
          EDA: parseFloat(EDA),
          HR: parseFloat(HR),
          TEMP: parseFloat(TEMP),
        };
      }

      const res = await axios.post('http://192.168.0.102:5000/predict', payload);
      setResponse(res.data);
    } catch (error) {
      Alert.alert('Error', error.response?.data?.error || 'An unexpected error occurred.');
    }
  };

  return (
    <ScrollView contentContainerStyle={styles.container}>
      <Text style={styles.title}>Health Prediction</Text>
      <View style={styles.switchContainer}>
        <Text style={styles.label}>Disease Prediction</Text>
        <Switch
          value={predictionType === 'stress'}
          onValueChange={togglePredictionType}
        />
        <Text style={styles.label}>Stress Prediction</Text>
      </View>

      {predictionType === 'disease' ? (
        <View style={styles.inputContainer}>
          <Text style={styles.label}>Enter Symptoms:</Text>
          <TextInput
            style={styles.textInput}
            placeholder="E.g., 10111001010001001 or I have cough"
            value={symptoms}
            onChangeText={setSymptoms}
          />
        </View>
      ) : (
        <View style={styles.inputContainer}>
          {['X', 'Y', 'Z', 'EDA', 'HR', 'TEMP'].map((field) => (
            <View key={field} style={styles.stressInputRow}>
              <Text style={styles.label}>{field}:</Text>
              <TextInput
                style={styles.textInput}
                placeholder={`Enter ${field}`}
                keyboardType="default"
                value={stressInputs[field]}
                onChangeText={(value) => handleStressInputChange(field, value)}
              />
            </View>
          ))}
        </View>
      )}

      <Button title="Submit" onPress={submitPrediction} />

      {response && (
        <View style={styles.responseContainer}>
          <Text style={styles.responseTitle}>Response:</Text>
          <Text style={styles.responseText}>{JSON.stringify(response, null, 2)}</Text>
        </View>
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flexGrow: 1,
    padding: 20,
    backgroundColor: '#f5f5f5',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    textAlign: 'center',
    marginBottom: 20,
  },
  switchContainer: {
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
    marginVertical: 20,
  },
  label: {
    fontSize: 16,
    marginHorizontal: 10,
  },
  inputContainer: {
    marginBottom: 20,
  },
  textInput: {
    borderWidth: 1,
    borderColor: '#ccc',
    borderRadius: 5,
    padding: 10,
    marginVertical: 10,
  },
  stressInputRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginVertical: 5,
  },
  responseContainer: {
    marginTop: 20,
    padding: 10,
    borderRadius: 5,
    backgroundColor: '#fff',
  },
  responseTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 10,
  },
  responseText: {
    fontSize: 14,
    color: '#333',
  },
});
