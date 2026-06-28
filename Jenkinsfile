pipeline {
  agent {
    kubernetes {
      yaml readFile('jenkins/k8s/agent-pod-template.yaml')
      defaultContainer 'python'
    }
  }

  triggers {
    githubPush()
  }

  parameters {
    string(name: 'ROUND', defaultValue: '', description: 'Attack round number (leave blank to auto-detect from results/)')
  }

  environment {
    PHISHING_PROJECT_PATH = '/workspace/phshing-email'
    RESULTS_DIR = 'results'
    HF_TOKEN = credentials('hf-token')
    GITHUB_TOKEN = credentials('github-token')
    GITHUB_REPO = credentials('github-repo')
    ADVERSARIAL_DATASET = 'Builder117/enterprise-adversarial-samples'
  }

  stages {

    stage('Detect Round') {
      steps {
        script {
          if (params.ROUND?.trim()) {
            env.ROUND = params.ROUND.trim()
          } else {
            env.ROUND = sh(
              script: '''python3 -c "
import json, os, glob
files = sorted(glob.glob('results/round_*_samples.json'))
if files:
    latest = files[-1]
    print(latest.split('_')[1])
else:
    import json as j
    mem = 'pipeline/attack_memory.json'
    if os.path.exists(mem):
        m = j.load(open(mem))
        rounds = m.get('rounds', [])
        print(rounds[-1]['round'] if rounds else 1)
    else:
        print(1)
"''',
              returnStdout: true
            ).trim()
          }
          echo "Round: ${env.ROUND}"
        }
      }
    }

    stage('Download Artifacts') {
      steps {
        sh """
          mkdir -p results agent_workspace agent_traces
          python pipeline/download_artifact.py --round ${env.ROUND}
        """
      }
    }

    stage('Attack Tests') {
      parallel {
        stage('Injection') {
          steps {
            sh "python pipeline/run_attacks.py --samples results/round_${env.ROUND}_samples.json --detector injection --round ${env.ROUND}"
          }
        }
        stage('Jailbreak') {
          steps {
            sh "python pipeline/run_attacks.py --samples results/round_${env.ROUND}_samples.json --detector jailbreak --round ${env.ROUND}"
          }
        }
        stage('Insecure Output') {
          steps {
            sh "python pipeline/run_attacks.py --samples results/round_${env.ROUND}_samples.json --detector insecure_output --round ${env.ROUND}"
          }
        }
        stage('Indirect Injection') {
          steps {
            sh "python pipeline/run_attacks.py --samples results/round_${env.ROUND}_samples.json --detector indirect_injection --round ${env.ROUND}"
          }
        }
      }
    }

    stage('Merge + Score') {
      steps {
        sh "python pipeline/merge_results.py --round ${env.ROUND}"
      }
    }

    stage('Orchestrator Agent') {
      steps {
        sh "python pipeline/trigger_agents.py --agent orchestrator --round ${env.ROUND}"

        script {
          def decision = readJSON file: 'results/pipeline_decision.json'
          env.AI_ACTION    = decision.action
          env.ARGO_WORKFLOW = decision.argo_workflow
          env.SEVERITY     = decision.severity
          echo "AI Decision: action=${env.AI_ACTION}, workflow=${env.ARGO_WORKFLOW}, severity=${env.SEVERITY}"
        }
      }
      post {
        always {
          archiveArtifacts artifacts: 'results/pipeline_decision.json, agent_traces/**', allowEmptyArchive: true
        }
      }
    }

    stage('Push Space Status') {
      steps {
        sh "python pipeline/push_space_status.py --round ${env.ROUND}"
      }
    }

    stage('Quality Gate') {
      steps {
        sh "pytest tests/test_guardrail.py -v --tb=short"
      }
    }

    stage('Scorecard') {
      steps {
        sh "python reporting/scorecard.py --round ${env.ROUND} --output results/scorecard.html"
      }
    }

    stage('Adversarial Retrain') {
      when {
        expression {
          return env.AI_ACTION == 'retrain' || env.AI_ACTION == 'partial_retrain'
        }
      }
      steps {
        sh """
          python pipeline/retrain.py \
            --decision results/pipeline_decision.json \
            --round ${env.ROUND}
        """
        sh "pytest tests/test_retrain.py -v --tb=short"
      }
    }

    stage('Download Retrained Models') {
      when {
        expression {
          return env.AI_ACTION == 'retrain' || env.AI_ACTION == 'partial_retrain'
        }
      }
      steps {
        sh "python pipeline/download_models.py --round ${env.ROUND}"
      }
      post {
        success {
          archiveArtifacts artifacts: "models/v${env.ROUND}/**", allowEmptyArchive: true
        }
      }
    }

    stage('Trigger Argo') {
      steps {
        withCredentials([string(credentialsId: 'argo-token', variable: 'ARGO_TOKEN')]) {
          sh """
            python pipeline/trigger_argo.py \
              --workflow ${env.ARGO_WORKFLOW} \
              --round ${env.ROUND} \
              --model-path models/v${env.ROUND}/
          """
        }
      }
    }

  }

  post {
    always {
      publishHTML(target: [
        allowMissing: true,
        alwaysLinkToLastBuild: true,
        keepAll: true,
        reportDir: 'results',
        reportFiles: 'scorecard.html',
        reportName: 'Adversarial Guardrail Report'
      ])
    }
    failure {
      echo "Pipeline failed. Check agent traces in archived artifacts."
    }
  }
}
