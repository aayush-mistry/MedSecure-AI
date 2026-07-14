import Fastify from 'fastify';
import fastifyCors from '@fastify/cors';
import fastifyJwt from '@fastify/jwt';
import fastifyMultipart from '@fastify/multipart';
import fastifyStatic from '@fastify/static';
import fastifyWebsocket from '@fastify/websocket';
import bcrypt from 'bcryptjs';
import fs from 'fs';
import path from 'path';
import 'dotenv/config';
import { fileURLToPath } from 'url';
import { z } from 'zod';
import { query, initDb } from './db.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PORT = parseInt(process.env.PORT || '3001', 10);
const ML_SERVICE_URLS = (process.env.ML_SERVICE_URLS || process.env.ML_SERVICE_URL || 'http://localhost:8000,http://127.0.0.1:8000,http://localhost:8001,http://127.0.0.1:8001')
  .split(',')
  .map(url => url.trim().replace(/\/+$/, ''))
  .filter(Boolean);
const JWT_SECRET = process.env.JWT_SECRET || 'medsecure-dev-secret';

const fastify = Fastify({ logger: true });

const uploadsDir = path.join(__dirname, 'uploads');
if (!fs.existsSync(uploadsDir)) {
  fs.mkdirSync(uploadsDir, { recursive: true });
}

await fastify.register(fastifyCors, { origin: '*' });
await fastify.register(fastifyJwt, { secret: JWT_SECRET });
await fastify.register(fastifyMultipart, { limits: { fileSize: 10 * 1024 * 1024 } });
await fastify.register(fastifyWebsocket);
await fastify.register(fastifyStatic, {
  root: uploadsDir,
  prefix: '/uploads/'
});

fastify.get('/', async () => {
  return {
    status: 'ok',
    service: 'MedSecure AI API',
    health: '/api/v1/health',
    frontend: 'http://localhost:5173'
  };
});

const wsClients = new Map();
const scanMessageCache = new Map();
const generateId = () => Math.random().toString(36).substring(2, 15) + Date.now().toString(36);

// ─── Zod validation schemas ───────────────────────────────────────────────────

const registerSchema = z.object({
  email: z.string().email('Invalid email address'),
  password: z.string().min(6, 'Password must be at least 6 characters'),
  role: z.enum(['consumer', 'pharmacist', 'healthcare_worker', 'inspector']),
  license_number: z.string().optional(),
  pin_code: z.string().optional()
});

const loginSchema = z.object({
  email: z.string().email('Invalid email address'),
  password: z.string().min(1, 'Password is required')
});

const reportSchema = z.object({
  medicine_id: z.string().min(1, 'medicine_id is required'),
  batch_number: z.string().min(1, 'batch_number is required'),
  lat: z.number().optional(),
  lng: z.number().optional(),
  notes: z.string().optional(),
  scan_id: z.string().optional()
});

function validate(schema, data) {
  const result = schema.safeParse(data);
  if (!result.success) {
    const error = result.error.errors.map(e => e.message).join(', ');
    return { valid: false, error };
  }
  return { valid: true, data: result.data };
}

async function authenticate(request, reply) {
  try {
    await request.jwtVerify();
  } catch (err) {
    reply.status(401).send({ error: 'Unauthorized' });
  }
}

async function optionalAuth(request) {
  try {
    await request.jwtVerify();
  } catch (err) {
    request.user = null;
  }
}

await initDb();

// ─── WebSocket — real-time scan progress ──────────────────────────────────────

fastify.register(async function (fastify) {
  fastify.get('/ws/scan', { websocket: true }, (socket, req) => {
    let currentScanId = null;

    socket.on('message', (message) => {
      try {
        const data = JSON.parse(message.toString());
        if (data.action === 'join' && data.scanId) {
          currentScanId = data.scanId;
          if (!wsClients.has(currentScanId)) {
            wsClients.set(currentScanId, []);
          }
          wsClients.get(currentScanId).push(socket);
          socket.send(JSON.stringify({ status: 'subscribed', scanId: currentScanId }));
          const cachedMessage = scanMessageCache.get(currentScanId);
          if (cachedMessage) {
            socket.send(JSON.stringify(cachedMessage));
          }
        }
      } catch (err) {
        console.error('WS parse error:', err.message);
      }
    });

    socket.on('close', () => {
      if (currentScanId && wsClients.has(currentScanId)) {
        const remaining = wsClients.get(currentScanId).filter(s => s !== socket);
        if (remaining.length === 0) wsClients.delete(currentScanId);
        else wsClients.set(currentScanId, remaining);
      }
    });
  });
});

// ─── AUTH ─────────────────────────────────────────────────────────────────────

fastify.post('/api/v1/auth/register', async (request, reply) => {
  const parsed = validate(registerSchema, request.body);
  if (!parsed.valid) return reply.status(400).send({ error: parsed.error });
  const { email, password, role, license_number, pin_code } = parsed.data;

  const existing = await query.get('SELECT id FROM users WHERE email = ?', [email]);
  if (existing) return reply.status(409).send({ error: 'Email already registered' });

  const id = 'usr-' + generateId();
  const hash = bcrypt.hashSync(password, 10);

  await query.run(
    'INSERT INTO users (id, email, password_hash, role, verified, license_number, pin_code) VALUES (?,?,?,?,?,?,?)',
    [id, email, hash, role, 1, license_number || null, pin_code || null]
  );

  const token = fastify.jwt.sign({ id, email, role, verified: 1 });
  return { token, user: { id, email, role, verified: 1 } };
});

fastify.post('/api/v1/auth/login', async (request, reply) => {
  const parsed = validate(loginSchema, request.body);
  if (!parsed.valid) return reply.status(400).send({ error: parsed.error });
  const { email, password } = parsed.data;

  const user = await query.get('SELECT * FROM users WHERE email = ?', [email]);
  if (!user || !bcrypt.compareSync(password, user.password_hash)) {
    return reply.status(401).send({ error: 'Invalid credentials' });
  }

  const token = fastify.jwt.sign({ id: user.id, email: user.email, role: user.role, verified: user.verified });
  return { token, user: { id: user.id, email: user.email, role: user.role, verified: user.verified } };
});

fastify.get('/api/v1/auth/me', { preHandler: authenticate }, async (request) => {
  return await query.get(
    'SELECT id,email,role,verified,license_number,pin_code,language FROM users WHERE id=?',
    [request.user.id]
  );
});

// ─── MEDICINES ────────────────────────────────────────────────────────────────

fastify.get('/api/v1/medicines/search', async (request) => {
  const q = request.query.q;
  if (!q) return [];
  const results = await query.all(
    `SELECT id, name, generic_name, manufacturer_name, composition, expected_colors, approved_batch_format, barcode_required
     FROM medicines WHERE name LIKE ? OR generic_name LIKE ? OR manufacturer_name LIKE ? LIMIT 20`,
    [`%${q}%`, `%${q}%`, `%${q}%`]
  );
  return results.map(r => ({
    ...r,
    composition: JSON.parse(r.composition),
    expected_colors: JSON.parse(r.expected_colors)
  }));
});

fastify.get('/api/v1/medicines/:id', async (request, reply) => {
  const row = await query.get('SELECT * FROM medicines WHERE id = ?', [request.params.id]);
  if (!row) return reply.status(404).send({ error: 'Not found' });
  return {
    ...row,
    composition: JSON.parse(row.composition),
    expected_colors: JSON.parse(row.expected_colors)
  };
});

// List batches for a medicine
fastify.get('/api/v1/medicines/:id/batches', async (request, reply) => {
  const med = await query.get('SELECT id FROM medicines WHERE id = ?', [request.params.id]);
  if (!med) return reply.status(404).send({ error: 'Medicine not found' });
  const batches = await query.all(
    `SELECT * FROM medicine_batches WHERE medicine_id = ? ORDER BY created_at DESC`,
    [request.params.id]
  );
  return batches;
});

// Medicine substitution suggestion
fastify.get('/api/v1/medicines/:id/alternatives', async (request) => {
  const med = await query.get('SELECT generic_name, composition FROM medicines WHERE id = ?', [request.params.id]);
  if (!med) return [];
  const alts = await query.all(
    `SELECT id, name, generic_name, manufacturer_name, composition, expected_colors
     FROM medicines WHERE generic_name = ? AND id != ? LIMIT 10`,
    [med.generic_name, request.params.id]
  );
  return alts.map(r => ({
    ...r,
    composition: JSON.parse(r.composition),
    expected_colors: JSON.parse(r.expected_colors)
  }));
});

// ─── BATCH LOOKUP ─────────────────────────────────────────────────────────────

// Look up a batch by batch_number (optionally scoped to medicine_id)
fastify.get('/api/v1/batches/lookup', async (request, reply) => {
  const { batch_number, medicine_id } = request.query;
  if (!batch_number) return reply.status(400).send({ error: 'batch_number query param required' });

  let row;
  if (medicine_id) {
    row = await query.get(
      `SELECT mb.*, m.name as medicine_name, m.generic_name, m.manufacturer_name
       FROM medicine_batches mb JOIN medicines m ON mb.medicine_id = m.id
       WHERE mb.batch_number = ? AND mb.medicine_id = ?`,
      [batch_number, medicine_id]
    );
  } else {
    row = await query.get(
      `SELECT mb.*, m.name as medicine_name, m.generic_name, m.manufacturer_name
       FROM medicine_batches mb JOIN medicines m ON mb.medicine_id = m.id
       WHERE mb.batch_number = ?`,
      [batch_number]
    );
  }

  if (!row) return reply.status(404).send({ found: false, error: 'Batch not found in genuine batch database' });
  return { found: true, batch: row };
});

fastify.get('/api/v1/batches/:id', async (request, reply) => {
  const row = await query.get(
    `SELECT mb.*, m.name as medicine_name, m.generic_name, m.manufacturer_name
     FROM medicine_batches mb JOIN medicines m ON mb.medicine_id = m.id
     WHERE mb.id = ?`,
    [request.params.id]
  );
  if (!row) return reply.status(404).send({ error: 'Batch not found' });
  return row;
});

// ─── SCANS ────────────────────────────────────────────────────────────────────

fastify.post('/api/v1/scans', { preHandler: optionalAuth }, async (request, reply) => {
  const data = await request.file();
  if (!data) return reply.status(400).send({ error: 'No image uploaded' });

  const lat = parseFloat(request.headers['x-latitude']) || (20 + Math.random() * 10);
  const lng = parseFloat(request.headers['x-longitude']) || (72 + Math.random() * 8);

  const scanId = 'scan-' + generateId();
  const ext = path.extname(data.filename) || '.jpg';
  const fileName = `${scanId}${ext}`;
  const filePath = path.join(uploadsDir, fileName);

  const writeStream = fs.createWriteStream(filePath);
  await new Promise((resolve, reject) => {
    data.file.pipe(writeStream);
    data.file.on('error', reject);
    writeStream.on('finish', resolve);
    writeStream.on('error', reject);
  });

  const stats = fs.statSync(filePath);
  if (stats.size === 0) {
    fs.unlinkSync(filePath);
    return reply.status(400).send({ error: 'Uploaded image is empty' });
  }

  const relativeUrl = `/uploads/${fileName}`;
  const userId = request.user ? request.user.id : null;

  await query.run(
    'INSERT INTO scans (id, user_id, image_url, lat, lng) VALUES (?,?,?,?,?)',
    [scanId, userId, relativeUrl, lat, lng]
  );

  sendWs(scanId, {
    status: 'stage',
    scanId,
    stage: 'upload_received',
    stageIndex: 0,
    totalStages: 12,
    progress: 0.02
  });

  runMlPipeline(scanId, filePath, relativeUrl, lat, lng);

  return { scanId, status: 'processing', image_url: relativeUrl };
});

function sendWs(scanId, msg) {
  scanMessageCache.set(scanId, msg);
  if (msg.status === 'completed' || msg.status === 'error') {
    setTimeout(() => scanMessageCache.delete(scanId), 5 * 60 * 1000);
  }
  if (wsClients.has(scanId)) {
    wsClients.get(scanId).forEach(s => { try { s.send(JSON.stringify(msg)); } catch (e) {} });
  }
}

fastify.get('/api/v1/scan-status/:id', async (request, reply) => {
  const cachedMessage = scanMessageCache.get(request.params.id);
  if (!cachedMessage) {
    return reply.status(404).send({ error: 'No live status cached for this scan' });
  }
  return cachedMessage;
});

async function fetchWithTimeout(url, options = {}, timeoutMs = 8000) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } finally {
    clearTimeout(timeout);
  }
}

async function checkMlHealth() {
  const failures = [];
  for (const baseUrl of ML_SERVICE_URLS) {
    try {
      const response = await fetchWithTimeout(`${baseUrl}/health`, {}, 3000);
      if (response.ok) {
        return { available: true, url: baseUrl, detail: await response.json().catch(() => ({})) };
      }
      failures.push(`${baseUrl}/health -> ${response.status}`);
    } catch (err) {
      failures.push(`${baseUrl}/health -> ${err.message}`);
    }
  }
  return { available: false, urls: ML_SERVICE_URLS, failures };
}

fastify.get('/api/v1/ml/health', async (request, reply) => {
  const health = await checkMlHealth();
  if (!health.available) {
    return {
      status: 'unavailable',
      service: 'ml',
      message: 'ML service is not reachable. Start the ML server before running live scans.',
      tried: health.failures
    };
  }
  return { status: 'healthy', service: 'ml', url: health.url, detail: health.detail };
});

async function startMlScan(scanId, filePath) {
  const payload = JSON.stringify({ scan_id: scanId, file_path: filePath });
  const options = {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: payload
  };
  const endpoints = ['/process_scan', '/api/scan/upload'];
  const failures = [];

  for (const baseUrl of ML_SERVICE_URLS) {
    for (const endpoint of endpoints) {
      try {
        const response = await fetchWithTimeout(`${baseUrl}${endpoint}`, options, 10000);
        if (response.ok) return response;
        failures.push(`${baseUrl}${endpoint} -> ${response.status}`);
        if (response.status !== 404) break;
      } catch (err) {
        failures.push(`${baseUrl}${endpoint} -> ${err.message}`);
        break;
      }
    }
  }

  throw new Error(`ML pipeline failed to start. Tried: ${failures.join('; ')}`);
}

async function fetchMlProgress(scanId) {
  const endpoints = [`/scan_progress/${scanId}`, `/api/scan/${scanId}/progress`];

  for (const baseUrl of ML_SERVICE_URLS) {
    for (const endpoint of endpoints) {
      try {
        const response = await fetchWithTimeout(`${baseUrl}${endpoint}`, {}, 5000);
        if (response.ok) return response;
        if (response.status !== 404) {
          throw new Error(`ML progress endpoint ${baseUrl}${endpoint} returned status ${response.status}`);
        }
      } catch (err) {
        if (!['fetch failed', 'This operation was aborted'].includes(err.message)) {
          throw err;
        }
        break;
      }
    }
  }

  return null;
}

async function runMlPipeline(scanId, filePath, relativeUrl, lat, lng) {
  try {
    // 1. Start async ML pipeline
    await startMlScan(scanId, filePath);

    // 2. Poll progress and send real-time stage updates via WebSocket
    let lastStage = -1;
    let result = null;
    let missingProgressPolls = 0;
    const maxPolls = 600; // 60 second timeout

    for (let i = 0; i < maxPolls; i++) {
      await new Promise(r => setTimeout(r, 100));

      const progRes = await fetchMlProgress(scanId);
      if (!progRes) {
        missingProgressPolls += 1;
        if (missingProgressPolls >= 10) {
          throw new Error('ML accepted scan but no progress endpoint returned this scan ID');
        }
        continue;
      }
      missingProgressPolls = 0;

      const progress = await progRes.json();

      if (progress.status === 'error') {
        throw new Error(progress.error || 'ML pipeline error');
      }

      if (progress.stage_index !== undefined && progress.stage_index !== lastStage) {
        lastStage = progress.stage_index;
        sendWs(scanId, {
          status: 'stage',
          scanId,
          stage: progress.stage,
          stageIndex: progress.stage_index,
          totalStages: progress.total_stages,
          progress: progress.progress
        });
      }

      if (progress.status === 'complete' && progress.result) {
        result = progress.result;
        break;
      }
    }

    if (!result) throw new Error('ML pipeline timed out');

    const matchedMedicine = result.medicine_id
      ? await query.get(
          'SELECT name, generic_name, manufacturer_name FROM medicines WHERE id=?',
          [result.medicine_id]
        )
      : null;

    const verdict = result.authenticity_score >= 80 ? 'verified'
      : result.authenticity_score >= 55 ? 'caution' : 'high_risk';

    // Persist scan result with all new fields
    await query.run(
      `UPDATE scans SET
        medicine_id=?, batch_id=?, authenticity_score=?, verdict=?,
        ocr_extracted=?, db_match_results=?, image_analysis=?, barcode_status=?,
        anomalies=?, signal_breakdown=?, scanned_at=CURRENT_TIMESTAMP
       WHERE id=?`,
      [
        result.medicine_id || null,
        result.batch_id || null,
        result.authenticity_score,
        verdict,
        JSON.stringify(result.ocr_extracted || {}),
        JSON.stringify(result.db_match_results || {}),
        JSON.stringify(result.image_analysis || {}),
        JSON.stringify(result.barcode_status || {}),
        JSON.stringify(result.anomalies || []),
        JSON.stringify(result.breakdown || result.signal_breakdown || {}),
        scanId
      ]
    );

    // Auto-create counterfeit alert for high_risk scans
    const detectedBatch = result.ocr_extracted?.batch_number;
    if (verdict === 'high_risk' && result.medicine_id && detectedBatch && detectedBatch !== 'Not Detected') {
      const batch = detectedBatch;
      const existing = await query.get(
        'SELECT * FROM alerts WHERE medicine_id=? AND batch_number=?',
        [result.medicine_id, batch]
      );
      if (existing) {
        const newCount = existing.report_count + 1;
        await query.run(
          'UPDATE alerts SET report_count=?, severity=?, last_updated=CURRENT_TIMESTAMP WHERE id=?',
          [newCount, newCount >= 3 ? 'high' : 'caution', existing.id]
        );
      } else {
        await query.run(
          'INSERT INTO alerts (id,medicine_id,batch_number,report_count,lat,lng,severity) VALUES (?,?,?,1,?,?,?)',
          ['alt-' + generateId(), result.medicine_id, batch, lat, lng, 'caution']
        );
      }
    }

    sendWs(scanId, {
      status: 'completed',
      scanId,
      data: {
        id: scanId,
        image_url: relativeUrl,
        authenticity_score: result.authenticity_score,
        confidence: result.confidence,
        verdict,
        analysis_status: 'completed',
        analysis_mode: 'ml_pipeline',
        analysis_summary: 'Completed by the live ML inference pipeline.',
        ocr_extracted: result.ocr_extracted,
        ocr_field_details: result.ocr_field_details,
        raw_ocr_text: result.raw_ocr_text,
        db_match_results: result.db_match_results,
        image_analysis: result.image_analysis,
        barcode_status: result.barcode_status,
        anomalies: result.anomalies,
        signal_breakdown: result.signal_breakdown,
        medicine_id: result.medicine_id,
        batch_id: result.batch_id,
        medicine_name: result.medicine_name || matchedMedicine?.name || result.ocr_extracted?.name,
        generic_name: result.generic_name || matchedMedicine?.generic_name,
        manufacturer_name: result.manufacturer_name || matchedMedicine?.manufacturer_name,
        lat,
        lng
      }
    });

  } catch (err) {
    console.error(`ML pipeline error for ${scanId}:`, err.message);
    const fallbackBreakdown = {
      batch_number: null, manufacturing_date: null, expiry_date: null,
      manufacturer: null, medicine_name: null, image_analysis: 50, barcode: null
    };
    await query.run(
      `UPDATE scans SET verdict='caution', authenticity_score=50,
       ocr_extracted='{}', anomalies='["ML service unavailable - fallback score applied"]',
       signal_breakdown=?, db_match_results='{}', image_analysis='{}', barcode_status='{}' WHERE id=?`,
      [JSON.stringify(fallbackBreakdown), scanId]
    );
    sendWs(scanId, {
      status: 'completed',
      scanId,
      data: {
        id: scanId,
        authenticity_score: 50,
        verdict: 'caution',
        analysis_status: 'fallback',
        analysis_mode: 'backend_fallback',
        analysis_summary: `Live ML analysis did not complete: ${err.message}`,
        ocr_extracted: {},
        db_match_results: {},
        image_analysis: {},
        barcode_status: {},
        anomalies: ['ML service unavailable - fallback score applied'],
        signal_breakdown: fallbackBreakdown,
        lat,
        lng
      }
    });
  }
}

// Get full scan details
fastify.get('/api/v1/scans/:id', { preHandler: optionalAuth }, async (request, reply) => {
  const scan = await query.get(
    `SELECT s.*,
            m.name as medicine_name, m.generic_name, m.manufacturer_name,
            mb.batch_number as batch_batch_number, mb.manufacturing_date as batch_mfg_date,
            mb.expiry_date as batch_exp_date, mb.mrp as batch_mrp,
            mb.manufacturing_license as batch_license, mb.pack_type, mb.pack_size,
            mb.country_of_origin, mb.status as batch_status
     FROM scans s
     LEFT JOIN medicines m ON s.medicine_id = m.id
     LEFT JOIN medicine_batches mb ON s.batch_id = mb.id
     WHERE s.id=?`,
    [request.params.id]
  );
  if (!scan) return reply.status(404).send({ error: 'Scan not found' });

  return {
    ...scan,
    ocr_extracted: scan.ocr_extracted ? JSON.parse(scan.ocr_extracted) : null,
    db_match_results: scan.db_match_results ? JSON.parse(scan.db_match_results) : null,
    image_analysis: scan.image_analysis ? JSON.parse(scan.image_analysis) : null,
    barcode_status: scan.barcode_status ? JSON.parse(scan.barcode_status) : null,
    anomalies: scan.anomalies ? JSON.parse(scan.anomalies) : [],
    signal_breakdown: scan.signal_breakdown ? JSON.parse(scan.signal_breakdown) : null
  };
});

fastify.get('/api/v1/scans/history', { preHandler: authenticate }, async (request) => {
  return await query.all(
    `SELECT s.id, s.image_url, s.authenticity_score, s.verdict, s.scanned_at,
     m.name as medicine_name, m.generic_name, m.manufacturer_name
     FROM scans s LEFT JOIN medicines m ON s.medicine_id=m.id
     WHERE s.user_id=? ORDER BY s.scanned_at DESC LIMIT 50`,
    [request.user.id]
  );
});

// ─── ALERTS ───────────────────────────────────────────────────────────────────

fastify.get('/api/v1/alerts/map', async () => {
  const alerts = await query.all(
    `SELECT a.*, m.name as medicine_name, m.manufacturer_name, m.generic_name
     FROM alerts a JOIN medicines m ON a.medicine_id=m.id`
  );
  return {
    type: 'FeatureCollection',
    features: alerts.map(a => ({
      type: 'Feature', id: a.id,
      geometry: { type: 'Point', coordinates: [a.lng, a.lat] },
      properties: {
        medicine_name: a.medicine_name, manufacturer_name: a.manufacturer_name,
        batch_number: a.batch_number, report_count: a.report_count, severity: a.severity,
        generic_name: a.generic_name
      }
    }))
  };
});

fastify.get('/api/v1/alerts/feed', async () => {
  return await query.all(
    `SELECT a.*, m.name as medicine_name, m.generic_name, m.manufacturer_name
     FROM alerts a JOIN medicines m ON a.medicine_id=m.id ORDER BY a.last_updated DESC LIMIT 30`
  );
});

// ─── REPORTS ──────────────────────────────────────────────────────────────────

fastify.post('/api/v1/reports', { preHandler: authenticate }, async (request, reply) => {
  const parsed = validate(reportSchema, request.body);
  if (!parsed.valid) return reply.status(400).send({ error: parsed.error });
  const { medicine_id, batch_number, lat, lng, notes, scan_id } = parsed.data;

  // Update or create alert
  const existing = await query.get(
    'SELECT * FROM alerts WHERE medicine_id=? AND batch_number=?',
    [medicine_id, batch_number]
  );
  if (existing) {
    const c = existing.report_count + 1;
    await query.run(
      'UPDATE alerts SET report_count=?, severity=?, last_updated=CURRENT_TIMESTAMP WHERE id=?',
      [c, c >= 3 ? 'high' : 'caution', existing.id]
    );
  } else {
    await query.run(
      'INSERT INTO alerts (id,medicine_id,batch_number,report_count,lat,lng,severity) VALUES (?,?,?,1,?,?,?)',
      ['alt-' + generateId(), medicine_id, batch_number, lat || 22.0, lng || 73.0, 'caution']
    );
  }

  // Save report record
  await query.run(
    `INSERT INTO reports (id, scan_id, user_id, medicine_id, batch_number, notes, lat, lng)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
    ['rpt-' + generateId(), scan_id || null, request.user.id, medicine_id, batch_number, notes || null, lat || null, lng || null]
  );

  return { success: true };
});

// ─── DASHBOARD ────────────────────────────────────────────────────────────────

fastify.get('/api/v1/dashboard/pharmacist', { preHandler: authenticate }, async (request, reply) => {
  if (!['pharmacist', 'inspector'].includes(request.user.role)) {
    return reply.status(403).send({ error: 'Role not authorized' });
  }

  const total = await query.get('SELECT COUNT(*) as c FROM scans');
  const hr = await query.get("SELECT COUNT(*) as c FROM scans WHERE verdict='high_risk'");
  const ca = await query.get("SELECT COUNT(*) as c FROM scans WHERE verdict='caution'");
  const ve = await query.get("SELECT COUNT(*) as c FROM scans WHERE verdict='verified'");
  const al = await query.get('SELECT COUNT(*) as c FROM alerts');

  const recentScans = await query.all(
    `SELECT s.id, s.authenticity_score, s.verdict, s.scanned_at, s.image_url,
     m.name as medicine_name, m.manufacturer_name
     FROM scans s LEFT JOIN medicines m ON s.medicine_id=m.id
     ORDER BY s.scanned_at DESC LIMIT 15`
  );

  const topFlagged = await query.all(
    `SELECT m.name, m.manufacturer_name, COUNT(*) as flag_count
     FROM scans s JOIN medicines m ON s.medicine_id=m.id
     WHERE s.verdict='high_risk' GROUP BY m.name ORDER BY flag_count DESC LIMIT 5`
  );

  return {
    stats: { total_scans: total.c, high_risk: hr.c, caution: ca.c, verified: ve.c, active_alerts: al.c },
    recentScans,
    topFlagged
  };
});

fastify.get('/api/v1/analytics/district', { preHandler: authenticate }, async (request, reply) => {
  if (request.user.role !== 'inspector') return reply.status(403).send({ error: 'Inspector role required' });
  return await query.all(
    `SELECT m.manufacturer_name, s.verdict, COUNT(*) as count
     FROM scans s JOIN medicines m ON s.medicine_id=m.id GROUP BY m.manufacturer_name, s.verdict`
  );
});

fastify.get('/api/v1/health', async () => {
  return { status: 'healthy', db: 'sqlite', time: new Date().toISOString(), version: '3.0.0' };
});

// ─── Start server ─────────────────────────────────────────────────────────────

try {
  await fastify.listen({ port: PORT, host: '0.0.0.0' });
  console.log(`MedSecure Backend running → http://localhost:${PORT}`);
} catch (err) {
  fastify.log.error(err);
  process.exit(1);
}
