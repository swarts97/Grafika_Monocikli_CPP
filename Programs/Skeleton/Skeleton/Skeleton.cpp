//=============================================================================================
// Mintaprogram: Zöld háromszög. Ervenyes 2018. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjeloles kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Szoke Tibor Adam
// Neptun : GQ5E7S
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"

const char * const vertexSource = R"(
	#version 330
	precision highp float;

	uniform mat4 MVP;
	
	layout(location = 0) in vec2 vp;
	layout(location = 1) in vec3 vertexColor;	

	out vec3 color;

	void main() {
		color = vertexColor;
		gl_Position = vec4(vp.x, vp.y, 0, 1) * MVP;
	}
)";

const char * const fragmentSource = R"(
	#version 330
	precision highp float;
	
	in vec3 color;

	out vec4 outColor;

	void main() {
		outColor = vec4(color, 1);
	}
)";

const char * vertexSource2 = R"(
	#version 330
    precision highp float;

	uniform mat4 MVP;

	layout(location = 0) in vec2 vertexPosition;

	out vec2 texCoord;

	void main() {
		texCoord = (vertexPosition + vec2(1, 1)) / 2;
		gl_Position = vec4(vertexPosition.x, vertexPosition.y, 0, 1) * MVP;
	}
)";

const char * fragmentSource2 = R"(
	#version 330
    precision highp float;

	uniform sampler2D textureUnit;

	in vec2 texCoord;
	out vec4 fragmentColor;

	void main() {
		fragmentColor = texture(textureUnit, texCoord);
	}
)";

class Camera {
	vec2 center;
	vec2 size;
	bool follow = false;
public:
	Camera() : center(0, 0), size(2, 2) {}

	bool isFollowing() { return follow; }
	void changeFollowing() {
		if (follow == false)
			Zoom(0.5f);
		else {
			Zoom(2.0f);
			Follow(vec2(0.0f, 0.0f));
		}
		follow = !follow;
	}

	mat4 V() { return TranslateMatrix(-center); }
	mat4 P() { return ScaleMatrix(vec2(2 / size.x, 2 / size.y)); }

	mat4 Vinv() { return TranslateMatrix(center); }
	mat4 Pinv() { return ScaleMatrix(vec2(size.x / 2, size.y / 2)); }

	void Zoom(float s) { size = size * s; }
	void Pan(vec2 t) { center = center + t; }
	void Follow(vec2 t) { center = t; }
};

Camera camera;
GPUProgram gpuProgram;
GPUProgram bgProgram;

class Spline {
	GLuint				vao, vbo;
	std::vector<float>  vertexData;

	std::vector<float> cps;
	std::vector<float> ts;
	float tension = 0;

	float Hermite(float y0, float x0, float v0, float y1, float x1, float v1, float x) {
		float a0 = y0;
		float a1 = v0;
		float a2 = (y1 - y0) * (3 / pow((x1 - x0), 2)) - (v1 + v0 * 2) * (1 / (x1 - x0));
		float a3 = (y0 - y1) * (2 / pow((x1 - x0), 3)) + (v1 + v0) * (1 / pow((x1 - x0), 2));
		float y = a3 * pow((x - x0), 3) + a2 * pow((x - x0), 2) + a1 * (x - x0) + a0;
		return y;
	}

	float HermiteDerivate(float y0, float x0, float v0, float y1, float x1, float v1, float x) {
		float a0 = y0;
		float a1 = v0;
		float a2 = (y1 - y0) * (3 / pow((x1 - x0), 2)) - (v1 + v0 * 2) * (1 / (x1 - x0));
		float a3 = (y0 - y1) * (2 / pow((x1 - x0), 3)) + (v1 + v0) * (1 / pow((x1 - x0), 2));
		float y = 3.0f * a3 * pow((x - x0), 2) + 2.0f * a2 * (x - x0) + a1;
		return y;
	}

public:
	void setTension(float tens) { tension = tens; }
	float getTension() { return tension; }

	void Create() {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);
		glGenBuffers(1, &vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glEnableVertexAttribArray(0);
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), reinterpret_cast<void*>(0));
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), reinterpret_cast<void*>(2 * sizeof(float)));
	}

	void SetStartingField() {
		vertexData.clear();
		AddPoint(-1.5f, -0.6f);
		AddPoint(1.5f, -0.6f);

		AddPoint(-1.25f, -0.6f);
		AddPoint(-1.00f, -0.6f);

		AddPoint(1.25f, -0.6f);
		AddPoint(1.00f, -0.6f);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, vertexData.size() * sizeof(float), &vertexData[0], GL_DYNAMIC_DRAW);
	}

	void SetHills() {
		vertexData.clear();
		AddPoint(-1.0f, 0.3f);
		AddPoint(1.0f, 0.3f);

		AddPoint(-0.8f, 0.59f);
		AddPoint(-0.5f, 0.18f);
		AddPoint(-0.25f, 0.74f);
		AddPoint(0.22f, 0.43f);
		AddPoint(0.6f, 0.6f);
	}

	float Catmul(float x) {
		float v0 = 0.0f;
		float v1;
		for (unsigned int i = 0; i < cps.size(); i++) {
			if (ts[i] <= x && x <= ts[i + 1]) {
				if (i == 0) {
					v1 = (cps[i + 2] - cps[i + 1]) * (1.0f / (ts[i + 2] - ts[i + 1])) + (cps[i + 1] - cps[i]) * (1.0f / (ts[i + 1] - ts[i]));
					v1 = v1 * (1 - tension) * (0.5f);
					return Hermite(cps[i], ts[i], v0, cps[i + 1], ts[i + 1], v1, x);
				}
				else if (0 < i && i < cps.size() - 2) {
					v0 = (cps[i + 1] - cps[i]) * (1 / (ts[i + 1] - ts[i])) + (cps[i] - cps[i - 1]) * (1 / (ts[i] - ts[i - 1]));
					v0 = v0 * (1 - tension) * (0.5f);
					v1 = (cps[i + 2] - cps[i + 1]) * (1 / (ts[i + 2] - ts[i + 1])) + (cps[i + 1] - cps[i]) * (1 / (ts[i + 1] - ts[i]));
					v1 = v1 * (1 - tension) * (0.5f);
					return Hermite(cps[i], ts[i], v0, cps[i + 1], ts[i + 1], v1, x);
				}
				else {
					v0 = (cps[i + 1] - cps[i]) * (1 / (ts[i + 1] - ts[i])) + (cps[i] - cps[i - 1]) * (1 / (ts[i] - ts[i - 1]));
					v0 = v0 * (1 - tension) * (0.5f);
					v1 = (cps[i + 1] - cps[i]);
					v1 = v1 * (1 - tension) * (0.5f);
					return Hermite(cps[i], ts[i], v0, cps[i + 1], ts[i + 1], v1, x);
				}
			}
		}
	}

	float Derivate(float x) {
		float v0 = 0.0f;
		float v1;
		for (unsigned int i = 0; i < cps.size(); i++) {
			if (ts[i] <= x && x <= ts[i + 1]) {
				if (i == 0) {
					v1 = (cps[i + 2] - cps[i + 1]) * (1.0f / (ts[i + 2] - ts[i + 1])) + (cps[i + 1] - cps[i]) * (1.0f / (ts[i + 1] - ts[i]));
					v1 = v1 * (1 - tension) * (0.5f);
					return HermiteDerivate(cps[i], ts[i], v0, cps[i + 1], ts[i + 1], v1, x);
				}
				else if (0 < i && i < cps.size() - 2) {
					v0 = (cps[i + 1] - cps[i]) * (1 / (ts[i + 1] - ts[i])) + (cps[i] - cps[i - 1]) * (1 / (ts[i] - ts[i - 1]));
					v0 = v0 * (1 - tension) * (0.5f);
					v1 = (cps[i + 2] - cps[i + 1]) * (1 / (ts[i + 2] - ts[i + 1])) + (cps[i + 1] - cps[i]) * (1 / (ts[i + 1] - ts[i]));
					v1 = v1 * (1 - tension) * (0.5f);
					return HermiteDerivate(cps[i], ts[i], v0, cps[i + 1], ts[i + 1], v1, x);
				}
				else {
					v0 = (cps[i + 1] - cps[i]) * (1 / (ts[i + 1] - ts[i])) + (cps[i] - cps[i - 1]) * (1 / (ts[i] - ts[i - 1]));
					v0 = v0 * (1 - tension) * (0.5f);
					v1 = (cps[i + 1] - cps[i]);
					v1 = v1 * (1 - tension) * (0.5f);
					return HermiteDerivate(cps[i], ts[i], v0, cps[i + 1], ts[i + 1], v1, x);
				}
			}
		}
	}

	void ReCalculateMountain() {
		vertexData.clear();
		for (float x = -1.0f; x < 1.0f; x += 0.01f) {
			float y = Catmul(x);
			vertexLoader(x, y);
		}
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, vertexData.size() * sizeof(float), &vertexData[0], GL_DYNAMIC_DRAW);
	}

	void ReCalculateHill() {
		vertexData.clear();
		for (float x = -1.5f; x < 1.5f; x += 0.01f) {
			float y = Catmul(x);
			vertexLoader(x, y);
			vertexLoader(x, -3.0f);
		}
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, vertexData.size() * sizeof(float), &vertexData[0], GL_DYNAMIC_DRAW);
	}

	void vertexLoader(float x, float y) {
		vertexData.push_back(x);
		vertexData.push_back(y);
		vertexData.push_back(0);
		vertexData.push_back(0);
		vertexData.push_back(0);
	}

	void AddPoint(float cX, float cY) {
		if (cps.size() < 2) {
			cps.push_back(cY);
			ts.push_back(cX);
			vertexLoader(ts[ts.size() - 1], cps[cps.size() - 1]);
			vertexLoader(ts[ts.size() - 1], -1);
		}

		else {
			for (unsigned int i = 0; i < cps.size(); i++) {
				if (ts[i] < cX && cX < ts[i + 1]) {
					cps.insert(cps.begin() + i + 1, cY);
					ts.insert(ts.begin() + i + 1, cX);
					break;
				}
			}
			ReCalculateHill();
		}
	}

	void DrawHill() {
		if (vertexData.size() > 0) {
			mat4 MVPTransform = camera.V() * camera.P();
			MVPTransform.SetUniform(gpuProgram.getId(), "MVP");
			glBindVertexArray(vao);
			glDrawArrays(GL_TRIANGLE_STRIP, 0, vertexData.size() / 5);
		}
	}

	mat4 M() {
		return mat4(1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1);
	}

	void DrawMountain() {
		if (vertexData.size() > 0) {
			mat4 MVPTransform = M();
			MVPTransform.SetUniform(gpuProgram.getId(), "MVP");
			glBindVertexArray(vao);
			glDrawArrays(GL_LINE_STRIP, 0, vertexData.size() / 5);
		}
	}
};

class Background {
	GLuint vao, vbo;
	vec2 vertices[4];
	Texture *pTexture;
public:
	Background() {
		vertices[0] = vec2(-1.0f, -1.0f);
		vertices[1] = vec2(1.0f, -1.0f);
		vertices[2] = vec2(1.0f, 1.0f);
		vertices[3] = vec2(-1.0f, 1.0f);
	}
	void Create(Spline hegy) {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);
		glGenBuffers(1, &vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_DYNAMIC_DRAW);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);

		int width = 600;
		int height = 600;
		std::vector<vec4> image(width * height);
		float x_norm = 2.0f / width;
		float y_norm = 2.0f / height;
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				float brightness = y_norm * y / 2.0f;
				if (hegy.Catmul(x * x_norm - 1.0f) <= y * y_norm - 1.0f)
					image[y * width + x] = vec4(0, 0, 1, 1);
				else
					image[y * width + x] = vec4(brightness, brightness, brightness, 1);
			}
		}
		pTexture = new Texture(width, height, image);
	}

	mat4 M() {
		return mat4(1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1);
	}

	void Draw() {
		glBindVertexArray(vao);
		mat4 MVPTransform = M();
		MVPTransform.SetUniform(bgProgram.getId(), "MVP");
		pTexture->SetUniform(bgProgram.getId(), "textureUnit");
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	}
};

Background bg;

class Legs {
	GLuint				vao, vbo;
	std::vector<float>  vertexData;

	float radius = 0.04f;
	float comb = 0.06f;
	float sipcsont = 0.06f;

public:
	std::vector<vec4> legs;

	void Create() {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);
		glGenBuffers(1, &vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glEnableVertexAttribArray(0);
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), reinterpret_cast<void*>(0));
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), reinterpret_cast<void*>(2 * sizeof(float)));

		legComponents();
	}

	void legToVertex() {
		vertexData.clear();
		for (unsigned int i = 0; i < legs.size(); i++)
			vertexLoader(legs[i]);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, vertexData.size() * sizeof(float), &vertexData[0], GL_DYNAMIC_DRAW);
	}

	vec4 legCalculator(vec4 labfej, bool rightDir) {
		vec4 sipVec = vec4(legs[2].x - labfej.x, legs[2].y - labfej.y, 0.0f, 1.0f);
		sipVec = sipVec * (1 / sqrtf(sipVec.x * sipVec.x + sipVec.y * sipVec.y)) * sipcsont;
		vec4 combVec = vec4(labfej.x - legs[2].x, labfej.y - legs[2].y, 0.0f, 1.0f);
		combVec = combVec * (1 / sqrtf(combVec.x * combVec.x + combVec.y * combVec.y)) * comb;

		vec4 terdLentrol = labfej + sipVec;
		vec4 terdFentrol = legs[2] + combVec;

		while (true) {
			if (terdLentrol.x - terdFentrol.x > 0.01f || terdLentrol.y - terdFentrol.y > 0.01f)
				if (rightDir)
					combVec = combVec * Rotator(0.01f);
				else
					combVec = combVec * Rotator(-0.01f);
			else
				break;
			terdFentrol = legs[2] + combVec;
			if (terdLentrol.x - terdFentrol.x > 0.01f || terdLentrol.y - terdFentrol.y > 0.01f)
				if(rightDir)
					sipVec = sipVec * Rotator(-0.01f);
				else
					sipVec = sipVec * Rotator(0.01f);
			else
				break;
			terdLentrol = labfej + sipVec;
		}
		return terdLentrol;
	}

	void legComponents() {
		// 0 - ballábfej
		legs.push_back((vec4(-1.0f + radius / 2.0f, -0.6f + radius, 0.0f, 1.0f)));
		// 1 - baltérd
		legs.push_back(legCalculator((vec4(-1.0f + radius / 2.0f, -0.6f + radius, 0.0f, 1.0f)), true));
		// 2 - nyereg
		legs.push_back(vec4(-0.96f, -0.48f, 0.0f, 1.0f));
		// 3 - jobbtérd
		legs.push_back(legCalculator(vec4(-1.0f + 1.5f * radius, -0.6f + radius, 0.0f, 1.0f), true));
		// 4 - jobblábfej
		legs.push_back(vec4(-1.0f + 1.5f * radius, -0.6f + radius, 0.0f, 1.0f));

		legToVertex();
	}

	mat4 Rotator(float fi) {
		return mat4(cosf(fi), sinf(fi), 0, 0,
			-sinf(fi), cosf(fi), 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1);
	}

	mat4 Shifter(vec2 v) {
		return mat4(1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			v.x, v.y, 0, 1);
	}

	void vertexLoader(vec4 point) {
		vertexData.push_back(point.x);
		vertexData.push_back(point.y);
		vertexData.push_back(1);
		vertexData.push_back(1);
		vertexData.push_back(0);
	}

	void Animate(float fi, vec2 origo, bool rightDir) {
		vec2 eltolasiVektor = vec2(0.0f - origo.x, 0.0f - origo.y);

		if (rightDir) {
			legs[0] = legs[0] * Shifter(eltolasiVektor);
			legs[0] = legs[0] * Rotator(-fi / radius);
			legs[0] = legs[0] * Shifter(-eltolasiVektor);
			legs[1] = legCalculator(legs[0], rightDir);

			legs[4] = legs[4] * Shifter(eltolasiVektor);
			legs[4] = legs[4] * Rotator(-fi / radius);
			legs[4] = legs[4] * Shifter(-eltolasiVektor);
			legs[3] = legCalculator(legs[4], rightDir);
			legToVertex();
		}
		else {
			legs[0] = legs[0] * Shifter(eltolasiVektor);
			legs[0] = legs[0] * Rotator(fi / radius);
			legs[0] = legs[0] * Shifter(-eltolasiVektor);
			legs[1] = legCalculator(legs[0], rightDir);

			legs[4] = legs[4] * Shifter(eltolasiVektor);
			legs[4] = legs[4] * Rotator(fi / radius);
			legs[4] = legs[4] * Shifter(-eltolasiVektor);
			legs[3] = legCalculator(legs[4], rightDir);
			legToVertex();
		}
	}

	void DrawLegs() {
		if (vertexData.size() > 0) {
			mat4 MVPTransform = camera.V() * camera.P();
			MVPTransform.SetUniform(gpuProgram.getId(), "MVP");
			glBindVertexArray(vao);
			glDrawArrays(GL_LINE_STRIP, 0, vertexData.size() / 5);
		}
	}
};

Legs leg;

class Wheel {
	GLuint				vao, vbo;
	std::vector<float>  vertexData;

	std::vector<vec4> cycPoints;
	std::vector<vec4> seatAndMan;

	float v = 0.3f;
	float radius = 0.04f;
	float erint = -1.0f + radius;

public:
	bool rightDir = true;
	float getV() { return v; }

	vec2 getOrigo(){ return vec2(cycPoints[0].x, cycPoints[0].y); }

	void Create() {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);
		glGenBuffers(1, &vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glEnableVertexAttribArray(0);
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), reinterpret_cast<void*>(0));
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), reinterpret_cast<void*>(2 * sizeof(float)));

		wheelComponents();
	}

	vec4 eltol(vec4 pont, vec2 vektor, std::vector<vec4> &tarolo) {
		vec4 eltolt = pont * Shifter(vektor);
		tarolo.push_back(eltolt);
		return eltolt;
	}

	vec4 elforgat(vec4 mit, vec4 korul, float fi, std::vector<vec4> &tarolo) {
		vec4 retpoint;
		retpoint = mit * Shifter(vec2(0.0f - korul.x, 0.0f - korul.y));
		retpoint = retpoint * Rotator(fi);
		retpoint = retpoint * Shifter(vec2(korul.x - 0.0f, korul.y - 0.0f));

		tarolo.push_back(retpoint);
		return retpoint;
	}

	void cycToVertex() {
		vertexData.clear();
		for (unsigned int i = 0; i < cycPoints.size(); i++)
			vertexLoader(cycPoints[i]);
		for (unsigned int i = 0; i < seatAndMan.size(); i++)
			vertexLoader(seatAndMan[i]);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, vertexData.size() * sizeof(float), &vertexData[0], GL_DYNAMIC_DRAW);
	}

	void wheelComponents() {
		vertexData.clear();
		vec4 center = vec4(-1.0f + radius, -0.6f + radius, 0.0f, 1.0f);
		vec4 p = center;
		// cycPoints[0] az origo
			cycPoints.push_back(p);
		// Fuggolegesen le
			p = eltol(p, vec2(0.0f, -radius), cycPoints);
		// Fuggolegesen fel
			p = eltol(p, vec2(0.0f, 2.0f * radius), cycPoints);
		// Vissza origoba
			p = eltol(p, vec2(0.0f, -radius), cycPoints);
		// Majd jobbra
			p = eltol(p, vec2(-radius, 0.0f), cycPoints);
		// Origon at atmero
			p = eltol(p, vec2(2.0f * (center.x - p.x), 2.0f * (center.y - p.y)), cycPoints);
		// Origoba sugar
			p = eltol(p, vec2(center.x - p.x, center.y - p.y), cycPoints);
		// Jobbra le
			p = center * Shifter(vec2(0.0f, -radius));
			p = elforgat(p, center, M_PI / 4.0f, cycPoints);
		// Origon at atmero
			p = eltol(p, (vec2(2.0f * (center.x - p.x), 2.0f * (center.y - p.y))), cycPoints);
		// Origoba sugar
			p = eltol(p, vec2(center.x - p.x, center.y - p.y), cycPoints);
		// Jobbra fel
			p = center * Shifter(vec2(0.0f, radius));
			p = elforgat(p, center, M_PI / -4.0f, cycPoints);
		// Origon at atmero
			p = eltol(p, vec2(2.0f * (center.x - p.x), 2.0f * (center.y - p.y)), cycPoints);
		// Kör
			for (float i = 0.0f; i < M_PI * 2.0f; i += 0.005f)
				p = elforgat(p, center, 0.005f, cycPoints);
		// Origoba sugar
			p = eltol(p, vec2(center.x - p.x, center.y - p.y), cycPoints);
		// Fel
			p = eltol(p, vec2(0.0f, radius * 2.0f), seatAndMan);
		// Balra
			p = eltol(p, vec2(-0.02f, 0.0f), seatAndMan);
		// Jobbra
			p = eltol(p, vec2(0.04f, 0.0f), seatAndMan);
		// Vissza kozepre
			p = eltol(p, vec2(-0.02f, 0.0f), seatAndMan);
		// Nyak
			p = eltol(p, vec2(0.0f, 0.1f), seatAndMan);
		// Fej
		vec4 fejkozep = p * Shifter(vec2(0.0f, radius / 2.0f));
			for (float i = 0.0f; i <= M_PI * 2.0f; i += 0.01f)
				p = elforgat(p, fejkozep, 0.01f, seatAndMan);
		// Vissza a nyereghez
			p = eltol(p, vec2(0.0f, -0.1f), seatAndMan);
		cycToVertex();
	}

	mat4 Rotator(float fi) {
		return mat4(cosf(fi), sinf(fi), 0, 0,
			-sinf(fi), cosf(fi), 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1);
	}

	mat4 Shifter(vec2 v) {
		return mat4(1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			v.x, v.y, 0, 1);
	}

	void vertexLoader(vec4 point) {
		vertexData.push_back(point.x);
		vertexData.push_back(point.y);
		vertexData.push_back(1);
		vertexData.push_back(0);
		vertexData.push_back(0);
	}

	void wheelPosCalc(float ds, Spline hill) {
		float dx;
		float currentDeriv;
		float nextDeriv;
		vec2 normalvec;
		vec2 ujOrigo;
		vec2 eltolas;

		currentDeriv = hill.Derivate(erint);
		dx = ds * (1.0f / sqrtf(1.0f + currentDeriv * currentDeriv));
		nextDeriv = hill.Derivate(erint + dx);
		normalvec = vec2(-nextDeriv, 1) * (1.0f / sqrtf(1.0f + nextDeriv * nextDeriv)) * radius;
		ujOrigo = vec2(erint + dx, hill.Catmul(erint + dx)) + normalvec;
		eltolas = vec2(ujOrigo.x - cycPoints[0].x, ujOrigo.y - cycPoints[0].y);
		
		for (unsigned int i = 0; i < cycPoints.size(); i++)
			cycPoints[i] = cycPoints[i] * Shifter(eltolas);
		for (unsigned int i = 0; i < seatAndMan.size(); i++)
			seatAndMan[i] = seatAndMan[i] * Shifter(eltolas);
		for (unsigned int i = 0; i < leg.legs.size(); i++) {
			leg.legs[i] = leg.legs[i] * Shifter(eltolas);
		}
		leg.legToVertex();

		erint = erint + dx;
		cycToVertex();
	}

	void wheelVeloCalc(float x, Spline hill) {
		float F = 45.0f;
		float m = 8.0f;
		float g = 5.0f;
		float ro = 100.0;

		float deriv = hill.Derivate(x);
		deriv = deriv * (1.0f / sqrtf(1.0f + deriv * deriv));
		float alpha = asinf(deriv);

		if (rightDir)
			v = (F - m * g * sinf(alpha)) / ro;
		else 
			v = (F + m * g * sinf(alpha)) / ro;
	}

	void turnWheelPoints(float fi) {
		vec2 eltolasiVektor = vec2(0.0f - cycPoints[0].x, 0.0f - cycPoints[0].y);
		for (unsigned int i = 0; i < cycPoints.size(); i++) {
			cycPoints[i] = cycPoints[i] * Shifter(eltolasiVektor);
			cycPoints[i] = cycPoints[i] * Rotator(fi);
			cycPoints[i] = cycPoints[i] * Shifter(-eltolasiVektor);
		}
	}

	void Animate(float ds, Spline hill) {
		if (rightDir) {
			wheelPosCalc(ds, hill);
			wheelVeloCalc(erint, hill);
			turnWheelPoints(-ds / radius);
			if (fabs(1.0f - cycPoints[0].x) - radius < 0.01f)
				rightDir = false;
		}
		else {
			wheelPosCalc(-ds, hill);
			wheelVeloCalc(erint, hill);
			turnWheelPoints(ds / radius);
			if (fabs(-1.0f - cycPoints[0].x) - radius < 0.01f)
				rightDir = true;
		}
		if(camera.isFollowing() == true)
			camera.Follow(getOrigo());
	}

	void DrawWheel() {
		if (vertexData.size() > 0) {
			mat4 MVPTransform = camera.V() * camera.P();
			MVPTransform.SetUniform(gpuProgram.getId(), "MVP");
			glBindVertexArray(vao);
			glDrawArrays(GL_LINE_STRIP, 0, vertexData.size() / 5);
		}
	}
};

Spline hegy;
Spline domb;
Wheel cyc;

void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);

	hegy.Create();
	hegy.SetHills();
	hegy.setTension(0.5f);
	hegy.ReCalculateMountain();

	bg.Create(hegy);
	
	domb.Create();
	domb.SetStartingField();
	domb.setTension(-0.001f);

	cyc.Create();
	leg.Create();

	gpuProgram.Create(vertexSource, fragmentSource, "outColor");
	bgProgram.Create(vertexSource2, fragmentSource2, "fragmentColor");
}

void onDisplay() {
	glClearColor(0, 0, 1.0f, 0);
	glClear(GL_COLOR_BUFFER_BIT);

	bgProgram.Use();
	bg.Draw();

	gpuProgram.Use();
	domb.DrawHill();
	cyc.DrawWheel();
	leg.DrawLegs();

	glutSwapBuffers();
}

void onKeyboard(unsigned char key, int pX, int pY) {
	switch (key) {
		case ' ': camera.changeFollowing(); break;
	}
	domb.ReCalculateHill();
	glutPostRedisplay();
}

void onKeyboardUp(unsigned char key, int pX, int pY) {}

void onMouseMotion(int pX, int pY) {}

void onMouse(int button, int state, int pX, int pY) {
	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {
		float cX = 2.0f * pX / windowWidth - 1;
		float cY = 1.0f - 2.0f * pY / windowHeight;

		char * buttonStat;
		if (state == GLUT_DOWN) {
			buttonStat = "pressed";
		}
		if (state == GLUT_DOWN) {
			switch (button) {
			case GLUT_LEFT_BUTTON:   printf("Left button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY);
				break;
			}
		}
		domb.AddPoint(cX, cY);
	}
}

void onIdle() {
	static float tend = 0.0f;
	const float dt = 0.01f;
	float tstart = tend;
	tend = glutGet(GLUT_ELAPSED_TIME) / 1000.0f;
	for (float t = tstart; t < tend; t += dt) {
		float Dt = fmin(dt, tend - t);
			cyc.Animate(Dt * cyc.getV(), domb);
			leg.Animate(Dt * cyc.getV(), cyc.getOrigo(), cyc.rightDir);
	}
	glutPostRedisplay();
}